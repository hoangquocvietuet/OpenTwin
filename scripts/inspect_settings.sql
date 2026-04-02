-- Inspect persisted settings (these override .env defaults)
SELECT key, value
FROM app_settings
ORDER BY key;

