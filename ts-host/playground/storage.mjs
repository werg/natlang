const DATABASE = 'natlang-playground';
const VERSION = 1;
let openPromise;

function open() {
  if (!('indexedDB' in globalThis)) return Promise.reject(new Error('IndexedDB is unavailable'));
  return openPromise ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      for (const name of ['projects', 'runs', 'cases'])
        if (!database.objectStoreNames.contains(name)) database.createObjectStore(name, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function transaction(name, mode, operation) {
  const database = await open();
  return new Promise((resolve, reject) => {
    const tx = database.transaction(name, mode);
    const request = operation(tx.objectStore(name));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    tx.onerror = () => reject(tx.error);
  });
}

export const storage = {
  list: async name => transaction(name, 'readonly', store => store.getAll()),
  get: async (name, id) => transaction(name, 'readonly', store => store.get(id)),
  put: async (name, value) => transaction(name, 'readwrite', store => store.put(value)),
  delete: async (name, id) => transaction(name, 'readwrite', store => store.delete(id)),
};
