// Called only by the local Python server; stdout is captured in memory.
try {
  const { Entry } = require('@napi-rs/keyring');
  const token = new Entry('com.apify.cli', 'token').getPassword();
  if (token) process.stdout.write(token);
} catch { process.exitCode = 1; }
