/*
This file is used to bootstrap development database locally.

Note: ONLY development database;
*/
CREATE USER "soxes-ai-template" SUPERUSER;
CREATE DATABASE "soxes-ai-template" OWNER "soxes-ai-template" ENCODING 'utf-8';
-- For pgvector
\c "soxes-ai-template" "soxes-ai-template";
CREATE EXTENSION vector;
