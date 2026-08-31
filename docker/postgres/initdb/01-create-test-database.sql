-- Database `make test-integration` targets, kept separate from the
-- development database because the integration fixtures drop every table
-- between tests.
--
-- Provisioning only: the `vector` extension is created by the initial Alembic
-- migration, not here, so a database this script never touches (CI's service
-- container, a managed Postgres) is still fully set up by `alembic upgrade
-- head` alone. See docs/DATABASE.md.
CREATE DATABASE insurance_claims_test;
