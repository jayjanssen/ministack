"""Unit coverage for PostgreSQL TLS container launch; no Docker daemon needed."""

import copy
import datetime
import io
import tarfile
from unittest.mock import Mock, call

import pytest

from ministack.services import rds


TLS_ARGS = [
    "-c", "ssl=on",
    "-c", "ssl_cert_file=/ministack-rds-tls/server.crt",
    "-c", "ssl_key_file=/ministack-rds-tls/server.key",
]


@pytest.fixture(scope="session", autouse=True)
def reset_server():
    """Override the integration suite's HTTP reset for these isolated tests."""


@pytest.fixture(autouse=True)
def clear_tls_settings(monkeypatch):
    monkeypatch.delenv("MINISTACK_RDS_PG_SSL_CERT", raising=False)
    monkeypatch.delenv("MINISTACK_RDS_PG_SSL_KEY", raising=False)


@pytest.fixture
def docker_client():
    client = Mock()
    container = client.containers.create.return_value
    container.put_archive.return_value = True
    return client


@pytest.fixture
def container_kwargs():
    return {
        "image": "postgres:16-alpine",
        "name": "tls-unit-test",
        "detach": True,
        "environment": {"PGDATA": "/var/lib/postgresql/data/pgdata"},
        "volumes": {"existing-data": {"bind": "/var/lib/postgresql/data", "mode": "rw"}},
        "labels": {"ministack": "rds"},
        "ports": {"5432/tcp": 15432},
    }


@pytest.fixture
def pem_files(tmp_path, monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "source.crt"
    key_path = tmp_path / "source.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    monkeypatch.setenv("MINISTACK_RDS_PG_SSL_CERT", str(cert_path))
    monkeypatch.setenv("MINISTACK_RDS_PG_SSL_KEY", str(key_path))
    return cert_path, key_path


@pytest.mark.parametrize("engine", ["postgres", "aurora-postgresql"])
def test_unconfigured_postgres_passes_through(docker_client, container_kwargs, engine):
    original = copy.deepcopy(container_kwargs)
    result = rds._run_rds_container(docker_client, engine, container_kwargs)
    assert result is docker_client.containers.run.return_value
    docker_client.containers.run.assert_called_once_with(**original)
    docker_client.containers.create.assert_not_called()
    assert container_kwargs == original


@pytest.mark.parametrize("engine", ["mysql", "aurora-mysql"])
def test_mysql_ignores_tls_settings(monkeypatch, docker_client, container_kwargs, engine):
    monkeypatch.setenv("MINISTACK_RDS_PG_SSL_CERT", "/not/a/certificate")
    monkeypatch.setenv("MINISTACK_RDS_PG_SSL_KEY", "/not/a/key")
    result = rds._run_rds_container(docker_client, engine, container_kwargs)
    assert result is docker_client.containers.run.return_value
    docker_client.containers.run.assert_called_once_with(**container_kwargs)
    docker_client.containers.create.assert_not_called()


@pytest.mark.parametrize("engine", ["postgres", "aurora-postgresql"])
@pytest.mark.parametrize("cert,key", [("cert", None), (None, "key"), ("", "key"), ("cert", ""), ("", "")])
def test_tls_settings_must_be_paired(monkeypatch, docker_client, container_kwargs, engine, cert, key):
    for setting, value in (("CERT", cert), ("KEY", key)):
        if value is not None:
            monkeypatch.setenv(f"MINISTACK_RDS_PG_SSL_{setting}", value)
    with pytest.raises(rds._RdsPostgresTLSError, match="must both name PEM files"):
        rds._run_rds_container(docker_client, engine, container_kwargs)
    assert docker_client.mock_calls == []


@pytest.mark.parametrize("invalid", ["cert", "key", "mismatched", "missing"])
def test_invalid_pem_fails_before_create(pem_files, docker_client, container_kwargs, invalid):
    cert_path, key_path = pem_files
    if invalid == "cert":
        cert_path.write_bytes(b"invalid certificate")
    elif invalid == "key":
        key_path.write_bytes(b"invalid private key")
    elif invalid == "missing":
        key_path.unlink()
    else:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        key_path.write_bytes(other_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    with pytest.raises(rds._RdsPostgresTLSError, match="^PostgreSQL TLS container launch failed$"):
        rds._run_rds_container(docker_client, "postgres", container_kwargs)
    assert docker_client.mock_calls == []


@pytest.mark.parametrize("engine", ["postgres", "aurora-postgresql"])
def test_tls_archive_permissions_contents_and_start_order(pem_files, docker_client, container_kwargs, engine):
    original = copy.deepcopy(container_kwargs)
    container = rds._run_rds_container(docker_client, engine, container_kwargs)
    assert container is docker_client.containers.create.return_value
    created = docker_client.containers.create.call_args.kwargs
    assert created["user"] == "0:0"
    assert created["entrypoint"] == [
        "sh", "-c", 'chown -R postgres:postgres /ministack-rds-tls && exec "$@"',
        "ministack-pg-tls",
    ]
    assert created["command"] == ["docker-entrypoint.sh", "postgres"] + TLS_ARGS
    for field, value in original.items():
        assert created[field] == value
    assert container_kwargs == original
    destination, archive = container.put_archive.call_args.args
    assert destination == "/"
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        entries = bundle.getmembers()
        assert [entry.name for entry in entries] == [
            "ministack-rds-tls", "ministack-rds-tls/server.crt", "ministack-rds-tls/server.key",
        ]
        assert entries[0].isdir()
        assert entries[0].mode == 0o700
        for entry, source in zip(entries[1:], pem_files):
            assert entry.isfile()
            assert entry.mode == 0o600
            assert bundle.extractfile(entry).read() == source.read_bytes()
        for entry in entries:
            assert not ("/" + entry.name).startswith(original["environment"]["PGDATA"])
            for volume in original["volumes"].values():
                assert not ("/" + entry.name).startswith(volume["bind"])
    assert docker_client.mock_calls == [
        call.containers.create(**created),
        call.containers.create().put_archive("/", archive),
        call.containers.create().start(),
    ]
    for source in pem_files:
        assert str(source) not in repr(created)
        assert source.read_text() not in repr(created)


@pytest.mark.parametrize("failure", ["copy-rejected", "copy-error", "start-error"])
def test_failed_launch_removes_container_but_preserves_volumes(pem_files, docker_client, container_kwargs, failure):
    container = docker_client.containers.create.return_value
    if failure == "copy-rejected":
        container.put_archive.return_value = False
    elif failure == "copy-error":
        container.put_archive.side_effect = RuntimeError("sensitive Docker request body")
    else:
        container.start.side_effect = RuntimeError("sensitive Docker request body")
    with pytest.raises(rds._RdsPostgresTLSError, match="^PostgreSQL TLS container launch failed$") as exc:
        rds._run_rds_container(docker_client, "postgres", container_kwargs)
    assert exc.value.__suppress_context__
    container.remove.assert_called_once_with(force=True, v=False)
    docker_client.volumes.assert_not_called()
    assert docker_client.volumes.mock_calls == []
    if failure != "start-error":
        container.start.assert_not_called()
    else:
        container.start.assert_called_once_with()
    assert container.mock_calls[-1] == call.remove(force=True, v=False)
    docker_client.containers.run.assert_not_called()


def test_reader_bootstrap_receives_all_tls_arguments(pem_files, docker_client, container_kwargs):
    container_kwargs["entrypoint"] = ["/bin/bash"]
    container_kwargs["command"] = ["sh", "-c", rds._PG_READER_BOOTSTRAP_SCRIPT]
    original = copy.deepcopy(container_kwargs)
    rds._run_rds_container(docker_client, "aurora-postgresql", container_kwargs)
    command = docker_client.containers.create.call_args.kwargs["command"]
    assert command == [
        "/bin/bash", "sh", "-c", rds._PG_READER_BOOTSTRAP_SCRIPT,
        "ministack-pg-reader",  # sh -c's $0; ssl's first -c must remain in $@.
    ] + TLS_ARGS
    assert 'exec gosu postgres postgres "$@"' in rds._PG_READER_BOOTSTRAP_SCRIPT
    assert container_kwargs == original


@pytest.mark.parametrize("entrypoint", ["docker-entrypoint.sh", ["docker-entrypoint.sh"]])
def test_cold_image_pull_preserves_replication_arguments(pem_files, docker_client, container_kwargs, entrypoint):
    from docker.errors import ImageNotFound

    container_kwargs["entrypoint"] = entrypoint
    container_kwargs["command"] = [
        "postgres", "-c", "wal_level=replica", "-c", "max_wal_senders=10",
        "-c", "max_replication_slots=10", "-c", "hot_standby=on",
    ]
    original = copy.deepcopy(container_kwargs)
    container = docker_client.containers.create.return_value
    docker_client.containers.create.side_effect = [ImageNotFound("cold cache"), container]
    assert rds._run_rds_container(docker_client, "postgres", container_kwargs) is container
    created = docker_client.containers.create.call_args.kwargs
    assert created["command"] == ["docker-entrypoint.sh"] + original["command"] + TLS_ARGS
    assert docker_client.containers.create.call_args_list == [call(**created), call(**created)]
    archive = container.put_archive.call_args.args[1]
    assert docker_client.mock_calls == [
        call.containers.create(**created),
        call.images.pull(original["image"]),
        call.containers.create(**created),
        call.containers.create().put_archive("/", archive),
        call.containers.create().start(),
    ]
    assert container_kwargs == original
