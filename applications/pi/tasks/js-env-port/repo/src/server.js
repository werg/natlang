import { adminPort, loadConfig } from './config.js';

const config = loadConfig();
console.log(`listening on ${config.host}:${config.port}, admin on ${adminPort(config)}`);
