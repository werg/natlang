/** Server settings from environment variables. */
export function loadConfig(env = process.env) {
  return {
    host: env.HOST || 'localhost',
    port: env.PORT || 3000,
    debug: env.DEBUG === '1',
  };
}

/** The admin interface listens on the port after the main one. */
export function adminPort(config) {
  return config.port + 1;
}
