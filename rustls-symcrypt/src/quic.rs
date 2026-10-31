//! QUIC packet protection and header protection for the TLS 1.3 AES-GCM cipher suites.
//!
//! rustls drives QUIC through [`rustls::quic::Algorithm`]. For every TLS 1.3 suite whose
//! [`rustls::Tls13CipherSuite::quic`] field is `Some`, rustls derives the packet protection
//! key, the packet IV, and the header protection key (RFC 9001, Section 5.1) and asks the
//! algorithm for a [`rustls::quic::PacketKey`] and a [`rustls::quic::HeaderProtectionKey`].
//!
//! * Packet protection (RFC 9001, Section 5.3) is AES-GCM through [`GcmExpandedKey`], the same
//!   primitive the TLS 1.3 record layer in [`crate::tls13`] uses.
//! * Header protection (RFC 9001, Section 5.4.3) is a single AES-ECB block encryption of the
//!   16-byte sample. The `symcrypt` crate does not expose ECB, so the block is produced with
//!   AES-CBC and an all-zero chaining value: for a single block, CBC computes
//!   `AES(key, sample XOR 0)`, which is exactly the ECB encryption of the sample.
//!
//! `TLS13_CHACHA20_POLY1305_SHA256` keeps `quic: None`. Header protection for that suite needs
//! the raw ChaCha20 keystream (RFC 9001, Section 5.4.4), and the `symcrypt` crate only exposes
//! the ChaCha20-Poly1305 AEAD. rustls does not offer suites without QUIC support in QUIC
//! handshakes, so the ChaCha20 suite is simply skipped there.
use crate::cipher_suites::AesGcm;
use rustls::crypto::cipher::{AeadKey, Iv, Nonce};
use rustls::quic;
use rustls::Error;
use symcrypt::cipher::{AesExpandedKey, BlockCipherType, AES_BLOCK_SIZE};
use symcrypt::gcm::GcmExpandedKey;

/// Length of the AES-GCM authentication tag.
const TAG_LEN: usize = 16;

/// Length of the header protection sample, which is one AES block (RFC 9001, Section 5.4.3).
const SAMPLE_LEN: usize = AES_BLOCK_SIZE as usize;

/// Length of the header protection mask (RFC 9001, Section 5.4.1).
const MASK_LEN: usize = 5;

/// Largest plaintext AES-GCM may protect under a single nonce: 2^39 - 256 bits
/// (NIST SP 800-38D, Section 5.2.1.1).
const GCM_MAX_PLAINTEXT_LEN: u64 = (1 << 36) - 32;

/// [`quic::Algorithm`] for the AES-GCM TLS 1.3 suites.
///
/// `packet_alg` selects AES-128-GCM or AES-256-GCM and therefore the key length rustls derives.
/// The limits are the per-key packet counts from RFC 9001, Appendix B.1.
pub(crate) struct KeyBuilder {
    pub(crate) packet_alg: AesGcm,
    pub(crate) confidentiality_limit: u64,
    pub(crate) integrity_limit: u64,
}

impl quic::Algorithm for KeyBuilder {
    fn packet_key(&self, key: AeadKey, iv: Iv) -> Box<dyn quic::PacketKey> {
        debug_assert_eq!(key.as_ref().len(), self.packet_alg.key_size());
        Box::new(PacketKey::new(
            key.as_ref(),
            iv,
            self.confidentiality_limit,
            self.integrity_limit,
        ))
    }

    fn header_protection_key(&self, key: AeadKey) -> Box<dyn quic::HeaderProtectionKey> {
        debug_assert_eq!(key.as_ref().len(), self.packet_alg.key_size());
        Box::new(HeaderProtectionKey::new(key.as_ref()))
    }

    fn aead_key_len(&self) -> usize {
        self.packet_alg.key_size()
    }
}

/// [`quic::PacketKey`] backed by an AES-GCM [`GcmExpandedKey`].
///
/// `iv` is combined with the packet number (and, for multipath QUIC, the path ID) to form the
/// per-packet nonce (RFC 9001, Section 5.3).
pub(crate) struct PacketKey {
    key: GcmExpandedKey,
    iv: Iv,
    confidentiality_limit: u64,
    integrity_limit: u64,
}

impl PacketKey {
    fn new(key: &[u8], iv: Iv, confidentiality_limit: u64, integrity_limit: u64) -> Self {
        // `WrongKeySize` is the only error `GcmExpandedKey::new` can return: SymCrypt expands
        // the key into caller-provided storage without allocating. rustls always supplies a key
        // of `Algorithm::aead_key_len()` bytes (checked in `KeyBuilder`), so this cannot fail.
        Self {
            key: GcmExpandedKey::new(key, BlockCipherType::AesBlock).unwrap(),
            iv,
            confidentiality_limit,
            integrity_limit,
        }
    }

    /// Encrypts `payload` in place under `nonce`, authenticating `header`, and returns the tag.
    fn seal(&self, nonce: Nonce, header: &[u8], payload: &mut [u8]) -> Result<quic::Tag, Error> {
        if payload.len() as u64 > GCM_MAX_PLAINTEXT_LEN {
            return Err(Error::EncryptError);
        }

        let mut tag = [0u8; TAG_LEN];
        // This call cannot fail: the nonce and tag lengths are fixed and the payload length was
        // checked above.
        self.key
            .encrypt_in_place(&nonce.0, header, payload, &mut tag);
        Ok(quic::Tag::from(&tag[..]))
    }

    /// Decrypts `payload` (ciphertext followed by the tag) in place under `nonce`, authenticating
    /// `header`, and returns the plaintext prefix of `payload`.
    ///
    /// The tag is verified inside SymCrypt (`SymCryptGcmDecrypt` uses the side-channel-safe
    /// `SymCryptEqual`), so no secret-dependent comparison happens in this crate.
    fn open<'a>(
        &self,
        nonce: Nonce,
        header: &[u8],
        payload: &'a mut [u8],
    ) -> Result<&'a [u8], Error> {
        let plain_len = payload
            .len()
            .checked_sub(TAG_LEN)
            .ok_or(Error::DecryptError)?;
        let (ciphertext, tag) = payload.split_at_mut(plain_len);
        self.key
            .decrypt_in_place(&nonce.0, header, ciphertext, tag)
            .map_err(|_| Error::DecryptError)?;
        Ok(&payload[..plain_len])
    }
}

impl quic::PacketKey for PacketKey {
    fn encrypt_in_place(
        &self,
        packet_number: u64,
        header: &[u8],
        payload: &mut [u8],
    ) -> Result<quic::Tag, Error> {
        self.seal(Nonce::new(&self.iv, packet_number), header, payload)
    }

    fn encrypt_in_place_for_path(
        &self,
        path_id: u32,
        packet_number: u64,
        header: &[u8],
        payload: &mut [u8],
    ) -> Result<quic::Tag, Error> {
        self.seal(
            Nonce::for_path(path_id, &self.iv, packet_number),
            header,
            payload,
        )
    }

    fn decrypt_in_place<'a>(
        &self,
        packet_number: u64,
        header: &[u8],
        payload: &'a mut [u8],
    ) -> Result<&'a [u8], Error> {
        self.open(Nonce::new(&self.iv, packet_number), header, payload)
    }

    fn decrypt_in_place_for_path<'a>(
        &self,
        path_id: u32,
        packet_number: u64,
        header: &[u8],
        payload: &'a mut [u8],
    ) -> Result<&'a [u8], Error> {
        self.open(
            Nonce::for_path(path_id, &self.iv, packet_number),
            header,
            payload,
        )
    }

    fn tag_len(&self) -> usize {
        TAG_LEN
    }

    fn confidentiality_limit(&self) -> u64 {
        self.confidentiality_limit
    }

    fn integrity_limit(&self) -> u64 {
        self.integrity_limit
    }
}

/// [`quic::HeaderProtectionKey`] backed by an AES [`AesExpandedKey`] (RFC 9001, Section 5.4.3).
///
/// `symcrypt` 0.5.1 gives [`AesExpandedKey`] no `Drop` implementation, so unlike the GCM packet
/// key the expanded header protection key schedule is not wiped when this value is dropped.
/// Header protection only hides header fields (RFC 9001, Section 9.4), not payload contents, and
/// rust-symcrypt's main branch adds the missing wipe, so a future `symcrypt` bump closes the gap.
pub(crate) struct HeaderProtectionKey {
    key: AesExpandedKey,
}

impl HeaderProtectionKey {
    fn new(key: &[u8]) -> Self {
        // `WrongKeySize` is likewise the only error `AesExpandedKey::new` can return, and rustls
        // supplies `aead_key_len()` bytes (checked in `KeyBuilder`); see `PacketKey::new`.
        Self {
            key: AesExpandedKey::new(key).unwrap(),
        }
    }

    /// Computes the header protection mask `AES-ECB(hp_key, sample)[0..5]`.
    ///
    /// Fails without touching anything if `sample` is not exactly one AES block.
    fn mask(&self, sample: &[u8]) -> Result<[u8; MASK_LEN], Error> {
        let sample: &[u8; SAMPLE_LEN] = sample
            .try_into()
            .map_err(|_| Error::General("sample of invalid length".into()))?;

        // AES-CBC of one block with an all-zero chaining value is AES-ECB of that block.
        let mut chaining_value = [0u8; SAMPLE_LEN];
        let mut block = [0u8; SAMPLE_LEN];
        self.key
            .aes_cbc_encrypt(&mut chaining_value, sample, &mut block)
            .map_err(|symcrypt_error| {
                Error::General(format!("SymCryptError: {}", symcrypt_error))
            })?;

        let mut mask = [0u8; MASK_LEN];
        mask.copy_from_slice(&block[..MASK_LEN]);
        Ok(mask)
    }

    /// Applies or removes header protection (RFC 9001, Section 5.4.1).
    ///
    /// `masked` says whether `first` currently carries protection. The packet number length
    /// lives in the low two bits of the unprotected first byte, so it is read before masking
    /// and after unmasking.
    fn xor_in_place(
        &self,
        sample: &[u8],
        first: &mut u8,
        packet_number: &mut [u8],
        masked: bool,
    ) -> Result<(), Error> {
        let mask = self.mask(sample)?;

        // The mask has a fixed length, so `split_first` cannot fail.
        let (first_mask, pn_mask) = mask.split_first().unwrap();

        // A valid packet number is at most 4 bytes (RFC 9000, Section 17.1).
        if packet_number.len() > pn_mask.len() {
            return Err(Error::General("packet number too long".into()));
        }

        // Nothing has been modified up to this point; everything below is infallible.
        const LONG_HEADER_FORM: u8 = 0x80;
        let bits = match *first & LONG_HEADER_FORM == LONG_HEADER_FORM {
            true => 0x0f,  // Long header: the low 4 bits are protected.
            false => 0x1f, // Short header: the low 5 bits are protected.
        };

        let first_plain = match masked {
            true => *first ^ (first_mask & bits),
            false => *first,
        };
        let pn_len = (first_plain & 0x03) as usize + 1;

        *first ^= first_mask & bits;
        for (dst, m) in packet_number.iter_mut().zip(pn_mask).take(pn_len) {
            *dst ^= m;
        }

        Ok(())
    }
}

impl quic::HeaderProtectionKey for HeaderProtectionKey {
    fn encrypt_in_place(
        &self,
        sample: &[u8],
        first: &mut u8,
        packet_number: &mut [u8],
    ) -> Result<(), Error> {
        self.xor_in_place(sample, first, packet_number, false)
    }

    fn decrypt_in_place(
        &self,
        sample: &[u8],
        first: &mut u8,
        packet_number: &mut [u8],
    ) -> Result<(), Error> {
        self.xor_in_place(sample, first, packet_number, true)
    }

    fn sample_len(&self) -> usize {
        SAMPLE_LEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmac::HmacSha256;
    use crate::{TLS13_AES_128_GCM_SHA256, TLS13_AES_256_GCM_SHA384};
    use rustls::crypto::tls13::{expand, Hkdf, HkdfUsingHmac, OkmBlock};
    use rustls::quic::{HeaderProtectionKey as _, Keys, PacketKey as _, Version};
    use rustls::{Side, SupportedCipherSuite, Tls13CipherSuite};

    /// Decodes hex, ignoring whitespace, so RFC appendices can be pasted verbatim.
    fn from_hex(s: &str) -> Vec<u8> {
        let digits: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
        assert_eq!(digits.len() % 2, 0, "odd number of hex digits");
        digits
            .chunks(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn tls13(suite: &SupportedCipherSuite) -> &'static Tls13CipherSuite {
        suite.tls13().expect("TLS 1.3 suite")
    }

    fn quic_alg(suite: &SupportedCipherSuite) -> &'static dyn quic::Algorithm {
        tls13(suite).quic.expect("suite supports QUIC")
    }

    /// The limits used for both AES-GCM suites (RFC 9001, Appendix B.1).
    const AES_GCM_CONFIDENTIALITY_LIMIT: u64 = 1 << 23;
    const AES_GCM_INTEGRITY_LIMIT: u64 = 1 << 52;

    fn packet_key(key: &str, iv: &str) -> PacketKey {
        PacketKey::new(
            &from_hex(key),
            Iv::copy(&from_hex(iv)),
            AES_GCM_CONFIDENTIALITY_LIMIT,
            AES_GCM_INTEGRITY_LIMIT,
        )
    }

    fn header_key(key: &str) -> HeaderProtectionKey {
        HeaderProtectionKey::new(&from_hex(key))
    }

    // RFC 9001, Appendix A.1: the keys derived for the example Initial packets.
    const CLIENT_INITIAL_SECRET: &str =
        "c00cf151ca5be075ed0ebfb5c80323c42d6b7db67881289af4008f1f6c357aea";
    const CLIENT_KEY: &str = "1f369613dd76d5467730efcbe3b1a22d";
    const CLIENT_IV: &str = "fa044b2f42a3fd3b46fb255c";
    const CLIENT_HP: &str = "9f50449e04a0e810283a1e9933adedd2";
    const SERVER_KEY: &str = "cf3a5331653c364c88f0f379b6067e37";
    const SERVER_IV: &str = "0ac1493ca1905853b0bba03e";
    const SERVER_HP: &str = "c206b8d9b9f0f37644430b490eeaa314";

    /// HKDF-Expand-Label info values from RFC 9001, Appendix A.1.
    const QUIC_KEY_INFO: &str = "00100e746c7331332071756963206b657900";
    const QUIC_IV_INFO: &str = "000c0d746c733133207175696320697600";
    const QUIC_HP_INFO: &str = "00100d746c733133207175696320687000";

    /// Connection ID shared by the RFC 9001 Appendix A examples.
    const INITIAL_CID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

    /// Both Appendix A Initial headers have the packet number at offset 18.
    const PN_OFFSET: usize = 18;

    // RFC 9001, Appendix A.2: the client Initial packet, packet number 2, 4-byte encoding.
    const CLIENT_CRYPTO_FRAME: &str =
        "060040f1010000ed0303ebf8fa56f129 39b9584a3896472ec40bb863cfd3e868
         04fe3a47f06a2b69484c000004130113 02010000c000000010000e00000b6578
         616d706c652e636f6dff01000100000a 00080006001d00170018001000070005
         04616c706e0005000501000000000033 00260024001d00209370b2c9caa47fba
         baf4559fedba753de171fa71f50f1ce1 5d43e994ec74d748002b000302030400
         0d0010000e0403050306030203080408 050806002d00020101001c0002400100
         3900320408ffffffffffffffff050480 00ffff07048000ffff08011001048000
         75300901100f088394c8f03e51570806 048000ffff";
    const CLIENT_PAYLOAD_LEN: usize = 1162;
    const CLIENT_HEADER: &str = "c300000001088394c8f03e5157080000449e00000002";
    const CLIENT_PACKET_NUMBER: u64 = 2;
    const CLIENT_SAMPLE: &str = "d1b1c98dd7689fb8ec11d242b123dc9b";
    const CLIENT_MASK: &str = "437b9aec36";
    const CLIENT_PROTECTED_PACKET: &str = "
         c000000001088394c8f03e5157080000 449e7b9aec34d1b1c98dd7689fb8ec11
         d242b123dc9bd8bab936b47d92ec356c 0bab7df5976d27cd449f63300099f399
         1c260ec4c60d17b31f8429157bb35a12 82a643a8d2262cad67500cadb8e7378c
         8eb7539ec4d4905fed1bee1fc8aafba1 7c750e2c7ace01e6005f80fcb7df6212
         30c83711b39343fa028cea7f7fb5ff89 eac2308249a02252155e2347b63d58c5
         457afd84d05dfffdb20392844ae81215 4682e9cf012f9021a6f0be17ddd0c208
         4dce25ff9b06cde535d0f920a2db1bf3 62c23e596d11a4f5a6cf3948838a3aec
         4e15daf8500a6ef69ec4e3feb6b1d98e 610ac8b7ec3faf6ad760b7bad1db4ba3
         485e8a94dc250ae3fdb41ed15fb6a8e5 eba0fc3dd60bc8e30c5c4287e53805db
         059ae0648db2f64264ed5e39be2e20d8 2df566da8dd5998ccabdae053060ae6c
         7b4378e846d29f37ed7b4ea9ec5d82e7 961b7f25a9323851f681d582363aa5f8
         9937f5a67258bf63ad6f1a0b1d96dbd4 faddfcefc5266ba6611722395c906556
         be52afe3f565636ad1b17d508b73d874 3eeb524be22b3dcbc2c7468d54119c74
         68449a13d8e3b95811a198f3491de3e7 fe942b330407abf82a4ed7c1b311663a
         c69890f4157015853d91e923037c227a 33cdd5ec281ca3f79c44546b9d90ca00
         f064c99e3dd97911d39fe9c5d0b23a22 9a234cb36186c4819e8b9c5927726632
         291d6a418211cc2962e20fe47feb3edf 330f2c603a9d48c0fcb5699dbfe58964
         25c5bac4aee82e57a85aaf4e2513e4f0 5796b07ba2ee47d80506f8d2c25e50fd
         14de71e6c418559302f939b0e1abd576 f279c4b2e0feb85c1f28ff18f58891ff
         ef132eef2fa09346aee33c28eb130ff2 8f5b766953334113211996d20011a198
         e3fc433f9f2541010ae17c1bf202580f 6047472fb36857fe843b19f5984009dd
         c324044e847a4f4a0ab34f719595de37 252d6235365e9b84392b061085349d73
         203a4a13e96f5432ec0fd4a1ee65accd d5e3904df54c1da510b0ff20dcc0c77f
         cb2c0e0eb605cb0504db87632cf3d8b4 dae6e705769d1de354270123cb11450e
         fc60ac47683d7b8d0f811365565fd98c 4c8eb936bcab8d069fc33bd801b03ade
         a2e1fbc5aa463d08ca19896d2bf59a07 1b851e6c239052172f296bfb5e724047
         90a2181014f3b94a4e97d117b4381303 68cc39dbb2d198065ae3986547926cd2
         162f40a29f0c3c8745c0f50fba3852e5 66d44575c29d39a03f0cda721984b6f4
         40591f355e12d439ff150aab7613499d bd49adabc8676eef023b15b65bfc5ca0
         6948109f23f350db82123535eb8a7433 bdabcb909271a6ecbcb58b936a88cd4e
         8f2e6ff5800175f113253d8fa9ca8885 c2f552e657dc603f252e1a8e308f76f0
         be79e2fb8f5d5fbbe2e30ecadd220723 c8c0aea8078cdfcb3868263ff8f09400
         54da48781893a7e49ad5aff4af300cd8 04a6b6279ab3ff3afb64491c85194aab
         760d58a606654f9f4400e8b38591356f bf6425aca26dc85244259ff2b19c41b9
         f96f3ca9ec1dde434da7d2d392b905dd f3d1f9af93d1af5950bd493f5aa731b4
         056df31bd267b6b90a079831aaf579be 0a39013137aac6d404f518cfd4684064
         7e78bfe706ca4cf5e9c5453e9f7cfd2b 8b4c8d169a44e55c88d4a9a7f9474241
         e221af44860018ab0856972e194cd934";

    // RFC 9001, Appendix A.3: the server Initial packet, packet number 1, 2-byte encoding.
    const SERVER_PAYLOAD: &str = "02000000000600405a020000560303ee fce7f7b37ba1d1632e96677825ddf739
         88cfc79825df566dc5430b9a045a1200 130100002e00330024001d00209d3c94
         0d89690b84d08a60993c144eca684d10 81287c834d5311bcf32bb9da1a002b00
         020304";
    const SERVER_HEADER: &str = "c1000000010008f067a5502a4262b50040750001";
    const SERVER_PACKET_NUMBER: u64 = 1;
    const SERVER_SAMPLE: &str = "2cd0991cd25b0aac406a5816b6394100";
    const SERVER_MASK: &str = "2ec0d8356a";
    const SERVER_PROTECTED_PACKET: &str = "
         cf000000010008f067a5502a4262b500 4075c0d95a482cd0991cd25b0aac406a
         5816b6394100f37a1c69797554780bb3 8cc5a99f5ede4cf73c3ec2493a1839b3
         dbcba3f6ea46c5b7684df3548e7ddeb9 c3bf9c73cc3f3bded74b562bfb19fb84
         022f8ef4cdd93795d77d06edbb7aaf2f 58891850abbdca3d20398c276456cbc4
         2158407dd074ee";

    fn client_payload() -> Vec<u8> {
        let mut payload = from_hex(CLIENT_CRYPTO_FRAME);
        // Padded with PADDING frames (0x00) to 1162 bytes, per RFC 9001, Appendix A.2.
        payload.resize(CLIENT_PAYLOAD_LEN, 0);
        payload
    }

    /// Protects `header` and `payload` with `keys`, returning the complete packet.
    ///
    /// `header` must end with the packet number, encoded in `pn_len` bytes.
    fn protect(
        keys: &quic::DirectionalKeys,
        packet_number: u64,
        header: &[u8],
        payload: &[u8],
    ) -> Vec<u8> {
        let mut header = header.to_vec();
        let mut payload = payload.to_vec();
        let pn_len = (header[0] & 0x03) as usize + 1;
        let pn_offset = header.len() - pn_len;

        let tag = keys
            .packet
            .encrypt_in_place(packet_number, &header, &mut payload)
            .unwrap();
        payload.extend_from_slice(tag.as_ref());

        // The sample starts 4 bytes after the start of the packet number (RFC 9001, Section 5.4.2).
        let sample_offset = 4 - pn_len;
        let sample = payload[sample_offset..sample_offset + SAMPLE_LEN].to_vec();
        let (first, rest) = header.split_at_mut(1);
        keys.header
            .encrypt_in_place(&sample, &mut first[0], &mut rest[pn_offset - 1..])
            .unwrap();

        header.extend_from_slice(&payload);
        header
    }

    /// Removes protection from `packet` with `keys`, returning the header and the plaintext.
    fn unprotect(
        keys: &quic::DirectionalKeys,
        packet: &[u8],
        pn_offset: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut packet = packet.to_vec();
        let sample = packet[pn_offset + 4..pn_offset + 4 + SAMPLE_LEN].to_vec();
        let (first, rest) = packet.split_at_mut(1);
        keys.header
            .decrypt_in_place(
                &sample,
                &mut first[0],
                &mut rest[pn_offset - 1..pn_offset + 3],
            )
            .unwrap();

        let pn_len = (packet[0] & 0x03) as usize + 1;
        let (header, payload) = packet.split_at_mut(pn_offset + pn_len);
        let packet_number = header[pn_offset..]
            .iter()
            .fold(0u64, |acc, byte| (acc << 8) | u64::from(*byte));
        let plaintext = keys
            .packet
            .decrypt_in_place(packet_number, header, payload)
            .unwrap()
            .to_vec();
        (header.to_vec(), plaintext)
    }

    #[test]
    fn aes_gcm_suites_support_quic() {
        for (suite, key_len) in [
            (&TLS13_AES_128_GCM_SHA256, 16),
            (&TLS13_AES_256_GCM_SHA384, 32),
        ] {
            let alg = quic_alg(suite);
            assert_eq!(alg.aead_key_len(), key_len);

            // Derive the keys through rustls so each builder receives exactly `aead_key_len()`
            // bytes, as the `quic::Algorithm` contract promises; `AeadKey::from` only builds
            // 32-byte keys.
            let keys = Keys::initial(Version::V1, tls13(suite), alg, &INITIAL_CID, Side::Client);
            assert_eq!(keys.local.packet.tag_len(), TAG_LEN);
            assert_eq!(
                keys.local.packet.confidentiality_limit(),
                AES_GCM_CONFIDENTIALITY_LIMIT
            );
            assert_eq!(keys.local.packet.integrity_limit(), AES_GCM_INTEGRITY_LIMIT);
            assert_eq!(keys.local.header.sample_len(), SAMPLE_LEN);
        }
    }

    #[cfg(feature = "chacha")]
    #[test]
    fn chacha20_suite_opts_out_of_quic() {
        // Raw ChaCha20 is not available through the `symcrypt` crate, so the suite is not
        // offered in QUIC handshakes. See the module documentation.
        assert!(tls13(&crate::TLS13_CHACHA20_POLY1305_SHA256).quic.is_none());
    }

    #[test]
    fn hkdf_expand_label_vectors_match_rfc9001_a1() {
        // Checks the Appendix A.1 key derivation with this crate's HKDF, which also validates
        // the `*_INFO` constants used by the multipath test below.
        let hkdf = HkdfUsingHmac(&HmacSha256);
        let expander = hkdf.expander_for_okm(&OkmBlock::new(&from_hex(CLIENT_INITIAL_SECRET)));
        let key: [u8; 16] = expand(expander.as_ref(), &[&from_hex(QUIC_KEY_INFO)]);
        let iv: [u8; 12] = expand(expander.as_ref(), &[&from_hex(QUIC_IV_INFO)]);
        let hp: [u8; 16] = expand(expander.as_ref(), &[&from_hex(QUIC_HP_INFO)]);
        assert_eq!(key[..], from_hex(CLIENT_KEY)[..]);
        assert_eq!(iv[..], from_hex(CLIENT_IV)[..]);
        assert_eq!(hp[..], from_hex(CLIENT_HP)[..]);
    }

    #[test]
    fn header_protection_mask_matches_rfc9001() {
        assert_eq!(
            header_key(CLIENT_HP)
                .mask(&from_hex(CLIENT_SAMPLE))
                .unwrap()[..],
            from_hex(CLIENT_MASK)[..]
        );
        assert_eq!(
            header_key(SERVER_HP)
                .mask(&from_hex(SERVER_SAMPLE))
                .unwrap()[..],
            from_hex(SERVER_MASK)[..]
        );
    }

    #[test]
    fn header_protection_mask_is_aes_ecb() {
        // FIPS 197, Appendix C.1 (AES-128) and C.3 (AES-256): the mask is the first five bytes
        // of the raw block encryption, so the zero-IV CBC construction must reproduce ECB.
        let plaintext = from_hex("00112233445566778899aabbccddeeff");
        assert_eq!(
            header_key("000102030405060708090a0b0c0d0e0f")
                .mask(&plaintext)
                .unwrap()[..],
            from_hex("69c4e0d86a7b0430d8cdb78070b4c55a")[..MASK_LEN]
        );
        assert_eq!(
            header_key("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
                .mask(&plaintext)
                .unwrap()[..],
            from_hex("8ea2b7ca516745bfeafc49904b496089")[..MASK_LEN]
        );
    }

    #[test]
    fn header_protection_rejects_bad_sample_length_without_changes() {
        let hp = header_key(CLIENT_HP);
        for len in [0, 15, 17, 32] {
            let mut first = 0xc3;
            let mut pn = [0x00, 0x00, 0x00, 0x02];
            assert!(hp
                .encrypt_in_place(&vec![0u8; len], &mut first, &mut pn)
                .is_err());
            assert!(hp
                .decrypt_in_place(&vec![0u8; len], &mut first, &mut pn)
                .is_err());
            assert_eq!(first, 0xc3);
            assert_eq!(pn, [0x00, 0x00, 0x00, 0x02]);
        }
    }

    #[test]
    fn header_protection_rejects_long_packet_number_without_changes() {
        let hp = header_key(CLIENT_HP);
        let sample = from_hex(CLIENT_SAMPLE);
        let mut first = 0xc3;
        let mut pn = [1, 2, 3, 4, 5];
        assert!(hp.encrypt_in_place(&sample, &mut first, &mut pn).is_err());
        assert!(hp.decrypt_in_place(&sample, &mut first, &mut pn).is_err());
        assert_eq!(first, 0xc3);
        assert_eq!(pn, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn header_protection_masks_only_the_encoded_packet_number_length() {
        let hp = header_key(CLIENT_HP);
        let sample = from_hex(CLIENT_SAMPLE);
        let mask = from_hex(CLIENT_MASK);

        // Long headers protect the low 4 bits of the first byte; short headers the low 5 bits.
        for (form, bits) in [(0xc0u8, 0x0fu8), (0x40u8, 0x1fu8)] {
            for pn_len in 1..=4usize {
                let first_plain = form | (pn_len as u8 - 1);
                let pn_plain = [0x11, 0x22, 0x33, 0x44];

                let mut first = first_plain;
                let mut pn = pn_plain;
                hp.encrypt_in_place(&sample, &mut first, &mut pn).unwrap();
                assert_eq!(first, first_plain ^ (mask[0] & bits));
                for (i, byte) in pn.iter().enumerate() {
                    let expected = match i < pn_len {
                        true => pn_plain[i] ^ mask[1 + i],
                        false => pn_plain[i],
                    };
                    assert_eq!(*byte, expected, "form {form:#x} pn_len {pn_len} byte {i}");
                }

                hp.decrypt_in_place(&sample, &mut first, &mut pn).unwrap();
                assert_eq!(first, first_plain);
                assert_eq!(pn, pn_plain);
            }
        }
    }

    #[test]
    fn packet_key_matches_rfc9001_a2_with_raw_keys() {
        // Exercises the packet key and header key directly with the Appendix A.1 key material,
        // independent of rustls' key derivation.
        let keys = quic::DirectionalKeys {
            header: Box::new(header_key(CLIENT_HP)),
            packet: Box::new(packet_key(CLIENT_KEY, CLIENT_IV)),
        };
        let packet = protect(
            &keys,
            CLIENT_PACKET_NUMBER,
            &from_hex(CLIENT_HEADER),
            &client_payload(),
        );
        assert_eq!(packet, from_hex(CLIENT_PROTECTED_PACKET));

        let (header, plaintext) = unprotect(&keys, &packet, PN_OFFSET);
        assert_eq!(header, from_hex(CLIENT_HEADER));
        assert_eq!(plaintext, client_payload());
    }

    #[test]
    fn packet_key_matches_rfc9001_a3_with_raw_keys() {
        let keys = quic::DirectionalKeys {
            header: Box::new(header_key(SERVER_HP)),
            packet: Box::new(packet_key(SERVER_KEY, SERVER_IV)),
        };
        let packet = protect(
            &keys,
            SERVER_PACKET_NUMBER,
            &from_hex(SERVER_HEADER),
            &from_hex(SERVER_PAYLOAD),
        );
        assert_eq!(packet, from_hex(SERVER_PROTECTED_PACKET));

        let (header, plaintext) = unprotect(&keys, &packet, PN_OFFSET);
        assert_eq!(header, from_hex(SERVER_HEADER));
        assert_eq!(plaintext, from_hex(SERVER_PAYLOAD));
    }

    #[test]
    fn initial_keys_match_rfc9001_a2_client() {
        // End to end through rustls: initial secrets, HKDF labels, and this module.
        let suite = tls13(&TLS13_AES_128_GCM_SHA256);
        let alg = quic_alg(&TLS13_AES_128_GCM_SHA256);
        let client = Keys::initial(Version::V1, suite, alg, &INITIAL_CID, Side::Client);
        let packet = protect(
            &client.local,
            CLIENT_PACKET_NUMBER,
            &from_hex(CLIENT_HEADER),
            &client_payload(),
        );
        let expected = from_hex(CLIENT_PROTECTED_PACKET);
        assert_eq!(expected.len(), 1200);
        assert_eq!(packet, expected);

        // The server removes the protection with its remote keys.
        let server = Keys::initial(Version::V1, suite, alg, &INITIAL_CID, Side::Server);
        let (header, plaintext) = unprotect(&server.remote, &expected, PN_OFFSET);
        assert_eq!(header, from_hex(CLIENT_HEADER));
        assert_eq!(plaintext, client_payload());
    }

    #[test]
    fn initial_keys_match_rfc9001_a3_server() {
        let suite = tls13(&TLS13_AES_128_GCM_SHA256);
        let alg = quic_alg(&TLS13_AES_128_GCM_SHA256);
        let server = Keys::initial(Version::V1, suite, alg, &INITIAL_CID, Side::Server);
        let packet = protect(
            &server.local,
            SERVER_PACKET_NUMBER,
            &from_hex(SERVER_HEADER),
            &from_hex(SERVER_PAYLOAD),
        );
        let expected = from_hex(SERVER_PROTECTED_PACKET);
        assert_eq!(packet, expected);

        let client = Keys::initial(Version::V1, suite, alg, &INITIAL_CID, Side::Client);
        let (header, plaintext) = unprotect(&client.remote, &expected, PN_OFFSET);
        assert_eq!(header, from_hex(SERVER_HEADER));
        assert_eq!(plaintext, from_hex(SERVER_PAYLOAD));
    }

    #[test]
    fn initial_keys_match_quic_v2_server() {
        // https://www.rfc-editor.org/rfc/rfc9369.html#name-sample-packet-protection
        let suite = tls13(&TLS13_AES_128_GCM_SHA256);
        let alg = quic_alg(&TLS13_AES_128_GCM_SHA256);
        let server = Keys::initial(Version::V2, suite, alg, &INITIAL_CID, Side::Server);
        let packet = protect(
            &server.local,
            SERVER_PACKET_NUMBER,
            &from_hex("d16b3343cf0008f067a5502a4262b50040750001"),
            &from_hex(SERVER_PAYLOAD),
        );
        let expected = from_hex(
            "dc6b3343cf0008f067a5502a4262b500 4075d92faaf16f05d8a4398c47089698
             baeea26b91eb761d9b89237bbf872630 17915358230035f7fd3945d88965cf17
             f9af6e16886c61bfc703106fbaf3cb4c fa52382dd16a393e42757507698075b2
             c984c707f0a0812d8cd5a6881eaf21ce da98f4bd23f6fe1a3e2c43edd9ce7ca8
             4bed8521e2e140",
        );
        assert_eq!(packet, expected);

        let client = Keys::initial(Version::V2, suite, alg, &INITIAL_CID, Side::Client);
        let (_, plaintext) = unprotect(&client.remote, &expected, PN_OFFSET);
        assert_eq!(plaintext, from_hex(SERVER_PAYLOAD));
    }

    #[test]
    fn packet_key_matches_gcm_known_answers() {
        // AES-GCM test cases 1, 2, 13, and 14 from the GCM specification: zero key, zero IV, no
        // additional data. Packet number 0 leaves the IV unchanged, so the nonce is the IV.
        let aes_128 = packet_key(
            "00000000000000000000000000000000",
            "000000000000000000000000",
        );
        let aes_256 = packet_key(
            "0000000000000000000000000000000000000000000000000000000000000000",
            "000000000000000000000000",
        );

        let mut empty = [];
        let tag = aes_128.encrypt_in_place(0, &[], &mut empty).unwrap();
        assert_eq!(
            tag.as_ref(),
            &from_hex("58e2fccefa7e3061367f1d57a4e7455a")[..]
        );
        let tag = aes_256.encrypt_in_place(0, &[], &mut empty).unwrap();
        assert_eq!(
            tag.as_ref(),
            &from_hex("530f8afbc74536b9a963b4f1c4cb738b")[..]
        );

        // Decrypting a tag-only payload yields an empty plaintext.
        let mut tag_only = from_hex("58e2fccefa7e3061367f1d57a4e7455a");
        assert!(aes_128
            .decrypt_in_place(0, &[], &mut tag_only)
            .unwrap()
            .is_empty());
        let mut tag_only = from_hex("530f8afbc74536b9a963b4f1c4cb738b");
        assert!(aes_256
            .decrypt_in_place(0, &[], &mut tag_only)
            .unwrap()
            .is_empty());

        let mut block = [0u8; 16];
        let tag = aes_128.encrypt_in_place(0, &[], &mut block).unwrap();
        assert_eq!(block[..], from_hex("0388dace60b6a392f328c2b971b2fe78")[..]);
        assert_eq!(
            tag.as_ref(),
            &from_hex("ab6e47d42cec13bdf53a67b21257bddf")[..]
        );

        let mut block = [0u8; 16];
        let tag = aes_256.encrypt_in_place(0, &[], &mut block).unwrap();
        assert_eq!(block[..], from_hex("cea7403d4d606b6e074ec5d3baf39d18")[..]);
        assert_eq!(
            tag.as_ref(),
            &from_hex("d0d1c8a799996bf0265b98b5d48ab919")[..]
        );
    }

    #[test]
    fn packet_key_decrypts_only_untampered_packets() {
        let key = packet_key(SERVER_KEY, SERVER_IV);
        let header = from_hex(SERVER_HEADER);
        let plaintext = from_hex(SERVER_PAYLOAD);

        let mut payload = plaintext.clone();
        let tag = key
            .encrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut payload)
            .unwrap();
        payload.extend_from_slice(tag.as_ref());

        // Round trip.
        let mut copy = payload.clone();
        assert_eq!(
            key.decrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut copy)
                .unwrap(),
            &plaintext[..]
        );

        // Wrong packet number, so a wrong nonce.
        let mut copy = payload.clone();
        assert_eq!(
            key.decrypt_in_place(SERVER_PACKET_NUMBER + 1, &header, &mut copy)
                .unwrap_err(),
            Error::DecryptError
        );

        // Tampered header (additional data).
        let mut copy = payload.clone();
        let mut bad_header = header.clone();
        bad_header[1] ^= 0x01;
        assert_eq!(
            key.decrypt_in_place(SERVER_PACKET_NUMBER, &bad_header, &mut copy)
                .unwrap_err(),
            Error::DecryptError
        );

        // Tampered ciphertext and tampered tag.
        for index in [0, payload.len() - 1] {
            let mut copy = payload.clone();
            copy[index] ^= 0x80;
            assert_eq!(
                key.decrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut copy)
                    .unwrap_err(),
                Error::DecryptError
            );
        }

        // Shorter than a tag.
        let mut short = payload[..TAG_LEN - 1].to_vec();
        assert_eq!(
            key.decrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut short)
                .unwrap_err(),
            Error::DecryptError
        );

        // Exactly a tag: an empty plaintext authenticates, and a flipped bit does not.
        let mut empty = [];
        let tag = key
            .encrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut empty)
            .unwrap();
        let mut tag_only = tag.as_ref().to_vec();
        assert!(key
            .decrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut tag_only)
            .unwrap()
            .is_empty());
        tag_only[0] ^= 0x01;
        assert_eq!(
            key.decrypt_in_place(SERVER_PACKET_NUMBER, &header, &mut tag_only)
                .unwrap_err(),
            Error::DecryptError
        );
    }

    #[test]
    fn packet_key_for_path_matches_picoquic() {
        // Based on picoquic's `multipath_aead_test`, as used by rustls' own providers. The key
        // and IV come from the same HKDF labels rustls uses for 1-RTT packet keys.
        const SECRET: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 35, 26, 27, 28, 29, 30, 31,
        ];
        const PATH_ID: u32 = 2;
        const PN: u64 = 12345;
        const PAYLOAD: &[u8] = b"The quick brown fox jumps over the lazy dog";
        const HEADER: &[u8] = b"This is a test";
        const EXPECTED: &[u8] = &[
            123, 139, 232, 52, 136, 25, 201, 143, 250, 89, 87, 39, 37, 63, 0, 210, 220, 227, 186,
            140, 183, 251, 13, 203, 6, 116, 204, 100, 166, 64, 43, 185, 174, 85, 212, 163, 242,
            141, 24, 166, 62, 228, 187, 137, 248, 31, 152, 126, 240, 151, 79, 51, 253, 130, 43,
            114, 173, 234, 254,
        ];

        let hkdf = HkdfUsingHmac(&HmacSha256);
        let expander = hkdf.expander_for_okm(&OkmBlock::new(&SECRET));
        let aead_key: [u8; 16] = expand(expander.as_ref(), &[&from_hex(QUIC_KEY_INFO)]);
        let iv: [u8; 12] = expand(expander.as_ref(), &[&from_hex(QUIC_IV_INFO)]);
        let key = PacketKey::new(
            &aead_key,
            Iv::new(iv),
            AES_GCM_CONFIDENTIALITY_LIMIT,
            AES_GCM_INTEGRITY_LIMIT,
        );

        let mut buf = PAYLOAD.to_vec();
        let tag = key
            .encrypt_in_place_for_path(PATH_ID, PN, HEADER, &mut buf)
            .unwrap();
        buf.extend_from_slice(tag.as_ref());
        assert_eq!(buf, EXPECTED);

        assert_eq!(
            key.decrypt_in_place_for_path(PATH_ID, PN, HEADER, &mut buf)
                .unwrap(),
            PAYLOAD
        );

        // A different path yields a different nonce.
        assert_eq!(
            key.decrypt_in_place_for_path(PATH_ID + 1, PN, HEADER, &mut EXPECTED.to_vec())
                .unwrap_err(),
            Error::DecryptError
        );
    }

    #[test]
    fn packet_key_for_path_round_trips_and_path_zero_matches_single_path() {
        let key = packet_key(CLIENT_KEY, CLIENT_IV);
        let header = from_hex(CLIENT_HEADER);
        let plaintext = from_hex(SERVER_PAYLOAD);

        for path_id in [0u32, 1, 2, 0xaead, u32::MAX] {
            let mut buf = plaintext.clone();
            let tag = key
                .encrypt_in_place_for_path(path_id, CLIENT_PACKET_NUMBER, &header, &mut buf)
                .unwrap();
            buf.extend_from_slice(tag.as_ref());

            if path_id == 0 {
                // Path 0 uses the plain packet number nonce (RFC 9001, Section 5.3).
                let mut single = plaintext.clone();
                let single_tag = key
                    .encrypt_in_place(CLIENT_PACKET_NUMBER, &header, &mut single)
                    .unwrap();
                single.extend_from_slice(single_tag.as_ref());
                assert_eq!(buf, single);
            }

            assert_eq!(
                key.decrypt_in_place_for_path(path_id, CLIENT_PACKET_NUMBER, &header, &mut buf)
                    .unwrap(),
                &plaintext[..]
            );
        }
    }
}
