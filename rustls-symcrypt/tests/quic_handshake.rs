//! QUIC handshakes between a rustls client and server that both use the SymCrypt provider,
//! followed by packet protection round trips with the keys the handshake produced.
//!
//! No network is involved: the handshake bytes are shuttled between the two connections with
//! `write_hs` and `read_hs`, the same way a QUIC implementation would carry them in CRYPTO frames.
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;

use rustls::crypto::CryptoProvider;
use rustls::quic::{
    ClientConnection, Connection, DirectionalKeys, KeyChange, Keys, PacketKey, Secrets,
    ServerConnection, Version,
};
use rustls::{CipherSuite, ClientConfig, Error, RootCertStore, ServerConfig};

use rustls_symcrypt::{
    custom_symcrypt_provider, default_symcrypt_provider, TLS13_AES_128_GCM_SHA256,
    TLS13_AES_256_GCM_SHA384,
};

const CLIENT_PARAMS: &[u8] = b"client transport parameters";
const SERVER_PARAMS: &[u8] = b"server transport parameters";

fn certs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("certs")
}

fn server_config(provider: CryptoProvider) -> Arc<ServerConfig> {
    let certs = rustls_pemfile::certs(&mut BufReader::new(
        File::open(certs_dir().join("localhost.pem")).unwrap(),
    ))
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
    let key = rustls_pemfile::private_key(&mut BufReader::new(
        File::open(certs_dir().join("localhost.key")).unwrap(),
    ))
    .unwrap()
    .unwrap();

    Arc::new(
        ServerConfig::builder_with_provider(Arc::new(provider))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .unwrap(),
    )
}

fn client_config(provider: CryptoProvider) -> Arc<ClientConfig> {
    let roots = rustls_pemfile::certs(&mut BufReader::new(
        File::open(certs_dir().join("RootCA.pem")).unwrap(),
    ))
    .collect::<Result<Vec<_>, _>>()
    .unwrap();
    let mut root_store = RootCertStore::empty();
    root_store.add_parsable_certificates(roots);

    Arc::new(
        ClientConfig::builder_with_provider(Arc::new(provider))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(root_store)
            .with_no_client_auth(),
    )
}

/// A completed handshake and the key changes each side reported along the way.
struct Session {
    client: Connection,
    server: Connection,
    client_changes: Vec<KeyChange>,
    server_changes: Vec<KeyChange>,
}

/// Runs one `write_hs` on `from` and feeds the output to `to`.
///
/// Returns whether anything happened, so the caller can detect when the handshake is idle.
fn transfer(from: &mut Connection, to: &mut Connection, changes: &mut Vec<KeyChange>) -> bool {
    let mut buf = Vec::new();
    let change = from.write_hs(&mut buf);
    if !buf.is_empty() {
        to.read_hs(&buf).unwrap_or_else(|err| {
            panic!("read_hs failed: {err:?} (alert: {:?})", to.alert());
        });
    }
    let progressed = !buf.is_empty() || change.is_some();
    changes.extend(change);
    progressed
}

fn handshake(
    client_provider: CryptoProvider,
    server_provider: CryptoProvider,
    version: Version,
) -> Session {
    let mut client = Connection::Client(
        ClientConnection::new(
            client_config(client_provider),
            version,
            "localhost".try_into().unwrap(),
            CLIENT_PARAMS.to_vec(),
        )
        .unwrap(),
    );
    let mut server = Connection::Server(
        ServerConnection::new(
            server_config(server_provider),
            version,
            SERVER_PARAMS.to_vec(),
        )
        .unwrap(),
    );

    let mut client_changes = Vec::new();
    let mut server_changes = Vec::new();
    loop {
        // Alternating single steps mirrors how packets flow: each side's output at one
        // encryption level is delivered before it moves on to the next level.
        let client_progressed = transfer(&mut client, &mut server, &mut client_changes);
        let server_progressed = transfer(&mut server, &mut client, &mut server_changes);
        if !client_progressed && !server_progressed {
            break;
        }
    }

    assert!(!client.is_handshaking(), "client handshake did not finish");
    assert!(!server.is_handshaking(), "server handshake did not finish");
    assert_eq!(client.alert(), None);
    assert_eq!(server.alert(), None);
    assert_eq!(client.quic_transport_parameters(), Some(SERVER_PARAMS));
    assert_eq!(server.quic_transport_parameters(), Some(CLIENT_PARAMS));
    assert!(client.zero_rtt_keys().is_none());
    assert!(server.zero_rtt_keys().is_none());

    Session {
        client,
        server,
        client_changes,
        server_changes,
    }
}

/// Splits the key changes of one side into its handshake keys and its 1-RTT keys and secrets.
fn split_changes(changes: Vec<KeyChange>) -> (Keys, Keys, Secrets) {
    let mut changes = changes.into_iter();
    let handshake = match changes.next() {
        Some(KeyChange::Handshake { keys }) => keys,
        other => panic!(
            "expected handshake keys first, got {}",
            describe(other.as_ref())
        ),
    };
    let (one_rtt, next) = match changes.next() {
        Some(KeyChange::OneRtt { keys, next }) => (keys, next),
        other => panic!(
            "expected 1-RTT keys second, got {}",
            describe(other.as_ref())
        ),
    };
    assert!(changes.next().is_none(), "unexpected additional key change");
    (handshake, one_rtt, next)
}

fn describe(change: Option<&KeyChange>) -> &'static str {
    match change {
        Some(KeyChange::Handshake { .. }) => "handshake keys",
        Some(KeyChange::OneRtt { .. }) => "1-RTT keys",
        None => "nothing",
    }
}

/// Builds a short-header packet with a 4-byte connection ID and a 2-byte packet number.
fn short_header(packet_number: u64) -> Vec<u8> {
    const PN_LEN: u8 = 2;
    let mut header = vec![0x40 | (PN_LEN - 1), 0xaa, 0xbb, 0xcc, 0xdd];
    header.extend_from_slice(&u16::try_from(packet_number).unwrap().to_be_bytes());
    header
}

/// Offset of the packet number in a `short_header` packet.
const PN_OFFSET: usize = 5;

fn plaintext() -> Vec<u8> {
    (0u8..=255).collect()
}

/// Protects a packet with `sender` and returns the complete protected packet.
fn protect(sender: &DirectionalKeys, packet_number: u64) -> Vec<u8> {
    let mut header = short_header(packet_number);
    let pn_len = header.len() - PN_OFFSET;
    let mut payload = plaintext();
    let tag = sender
        .packet
        .encrypt_in_place(packet_number, &header, &mut payload)
        .unwrap();
    payload.extend_from_slice(tag.as_ref());

    // The sample starts 4 bytes after the start of the packet number (RFC 9001, Section 5.4.2).
    let sample_offset = 4 - pn_len;
    let sample = payload[sample_offset..sample_offset + sender.header.sample_len()].to_vec();
    let (first, rest) = header.split_at_mut(1);
    sender
        .header
        .encrypt_in_place(&sample, &mut first[0], &mut rest[PN_OFFSET - 1..])
        .unwrap();

    header.extend_from_slice(&payload);
    header
}

/// Removes header protection with `receiver` and returns the header and the protected payload.
fn unprotect_header(receiver: &DirectionalKeys, packet: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut packet = packet.to_vec();
    let sample = packet[PN_OFFSET + 4..PN_OFFSET + 4 + receiver.header.sample_len()].to_vec();
    let (first, rest) = packet.split_at_mut(1);
    receiver
        .header
        .decrypt_in_place(
            &sample,
            &mut first[0],
            &mut rest[PN_OFFSET - 1..PN_OFFSET + 3],
        )
        .unwrap();

    let pn_len = (packet[0] & 0x03) as usize + 1;
    let (header, payload) = packet.split_at(PN_OFFSET + pn_len);
    (header.to_vec(), payload.to_vec())
}

fn packet_number(header: &[u8]) -> u64 {
    header[PN_OFFSET..]
        .iter()
        .fold(0u64, |acc, byte| (acc << 8) | u64::from(*byte))
}

/// Protects a packet with `sender` and fully unprotects it with `receiver`.
fn round_trip(sender: &DirectionalKeys, receiver: &DirectionalKeys, pn: u64) {
    let packet = protect(sender, pn);
    let (header, mut payload) = unprotect_header(receiver, &packet);
    assert_eq!(header, short_header(pn));
    assert_eq!(packet_number(&header), pn);
    let decrypted = receiver
        .packet
        .decrypt_in_place(pn, &header, &mut payload)
        .unwrap();
    assert_eq!(decrypted, &plaintext()[..]);
}

/// Encrypts with `sender` and decrypts with `receiver`, using the header as-is (no header
/// protection), for key sets that only carry packet keys.
fn packet_round_trip(sender: &dyn PacketKey, receiver: &dyn PacketKey, pn: u64) {
    let header = short_header(pn);
    let mut payload = plaintext();
    let tag = sender.encrypt_in_place(pn, &header, &mut payload).unwrap();
    payload.extend_from_slice(tag.as_ref());
    assert_eq!(
        receiver
            .decrypt_in_place(pn, &header, &mut payload)
            .unwrap(),
        &plaintext()[..]
    );
}

/// Checks that a packet encrypted with `sender` is rejected by `receiver`.
fn assert_rejected(sender: &dyn PacketKey, receiver: &dyn PacketKey, pn: u64) {
    let header = short_header(pn);
    let mut payload = plaintext();
    let tag = sender.encrypt_in_place(pn, &header, &mut payload).unwrap();
    payload.extend_from_slice(tag.as_ref());
    assert_eq!(
        receiver
            .decrypt_in_place(pn, &header, &mut payload)
            .unwrap_err(),
        Error::DecryptError
    );
}

fn check_session(session: Session, expected_suite: CipherSuite) {
    let Session {
        client,
        server,
        client_changes,
        server_changes,
    } = session;
    assert_eq!(
        client.negotiated_cipher_suite().unwrap().suite(),
        expected_suite
    );
    assert_eq!(
        server.negotiated_cipher_suite().unwrap().suite(),
        expected_suite
    );

    let (client_hs, client_1rtt, mut client_next) = split_changes(client_changes);
    let (server_hs, server_1rtt, mut server_next) = split_changes(server_changes);

    // Handshake-level keys agree in both directions.
    round_trip(&client_hs.local, &server_hs.remote, 0);
    round_trip(&server_hs.local, &client_hs.remote, 1);

    // So do the 1-RTT keys.
    round_trip(&client_1rtt.local, &server_1rtt.remote, 2);
    round_trip(&server_1rtt.local, &client_1rtt.remote, 3);

    // Handshake keys must not unprotect 1-RTT packets and vice versa.
    assert_rejected(&*client_hs.local.packet, &*server_1rtt.remote.packet, 4);
    assert_rejected(&*client_1rtt.local.packet, &*server_hs.remote.packet, 5);

    // Key updates stay in sync on both sides and invalidate the previous generation.
    let previous = (client_1rtt, server_1rtt);
    let mut previous_packet_keys = (previous.0.local.packet, previous.1.remote.packet);
    for generation in 1..=3u64 {
        let client_updated = client_next.next_packet_keys();
        let server_updated = server_next.next_packet_keys();
        let pn = 100 * generation;
        packet_round_trip(&*client_updated.local, &*server_updated.remote, pn);
        packet_round_trip(&*server_updated.local, &*client_updated.remote, pn + 1);
        assert_rejected(&*previous_packet_keys.0, &*server_updated.remote, pn + 2);
        assert_rejected(&*client_updated.local, &*previous_packet_keys.1, pn + 3);
        previous_packet_keys = (client_updated.local, server_updated.remote);
    }
}

#[test]
fn quic_v1_handshake_with_default_provider() {
    let session = handshake(
        default_symcrypt_provider(),
        default_symcrypt_provider(),
        Version::V1,
    );
    check_session(session, CipherSuite::TLS13_AES_256_GCM_SHA384);
}

#[test]
fn quic_v2_handshake_with_default_provider() {
    let session = handshake(
        default_symcrypt_provider(),
        default_symcrypt_provider(),
        Version::V2,
    );
    check_session(session, CipherSuite::TLS13_AES_256_GCM_SHA384);
}

#[test]
fn quic_handshake_negotiates_aes_128_gcm() {
    // The server only offers AES-128-GCM, so the client has to pick it.
    let session = handshake(
        default_symcrypt_provider(),
        custom_symcrypt_provider(Some(vec![TLS13_AES_128_GCM_SHA256]), None),
        Version::V1,
    );
    check_session(session, CipherSuite::TLS13_AES_128_GCM_SHA256);
}

#[test]
fn quic_handshake_with_aes_256_gcm_only() {
    let session = handshake(
        custom_symcrypt_provider(Some(vec![TLS13_AES_256_GCM_SHA384]), None),
        custom_symcrypt_provider(Some(vec![TLS13_AES_256_GCM_SHA384]), None),
        Version::V1,
    );
    check_session(session, CipherSuite::TLS13_AES_256_GCM_SHA384);
}

#[cfg(feature = "chacha")]
#[test]
fn quic_rejects_provider_with_only_chacha20() {
    // ChaCha20-Poly1305 opts out of QUIC (see the crate documentation), so a provider offering
    // only that suite cannot start a QUIC connection.
    use rustls_symcrypt::TLS13_CHACHA20_POLY1305_SHA256;

    let provider = || custom_symcrypt_provider(Some(vec![TLS13_CHACHA20_POLY1305_SHA256]), None);

    let client = ClientConnection::new(
        client_config(provider()),
        Version::V1,
        "localhost".try_into().unwrap(),
        CLIENT_PARAMS.to_vec(),
    );
    assert!(
        matches!(&client, Err(Error::General(msg)) if msg.contains("ciphersuite must support QUIC")),
        "unexpected result: {:?}",
        client.err()
    );

    let server = ServerConnection::new(
        server_config(provider()),
        Version::V1,
        SERVER_PARAMS.to_vec(),
    );
    assert!(
        matches!(&server, Err(Error::General(msg)) if msg.contains("ciphersuite must support QUIC")),
        "unexpected result: {:?}",
        server.err()
    );
}

#[cfg(feature = "chacha")]
#[test]
fn quic_handshake_skips_chacha20_even_when_preferred() {
    // Both sides prefer ChaCha20-Poly1305 for TLS, but QUIC handshakes only offer suites with
    // QUIC support, so AES-128-GCM is negotiated.
    use rustls_symcrypt::TLS13_CHACHA20_POLY1305_SHA256;

    let provider = || {
        custom_symcrypt_provider(
            Some(vec![
                TLS13_CHACHA20_POLY1305_SHA256,
                TLS13_AES_128_GCM_SHA256,
            ]),
            None,
        )
    };
    let session = handshake(provider(), provider(), Version::V1);
    check_session(session, CipherSuite::TLS13_AES_128_GCM_SHA256);
}
