/** Browser-owned durable state and traces. A commit is one IndexedDB transaction. */
export class StudioStore {
    static async open(name = 'natlang-studio-v1') {
        const request = indexedDB.open(name, 3);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains('states'))
                db.createObjectStore('states');
            if (!db.objectStoreNames.contains('history'))
                db.createObjectStore('history', { keyPath: 'id', autoIncrement: true });
            if (!db.objectStoreNames.contains('drafts'))
                db.createObjectStore('drafts');
            if (!db.objectStoreNames.contains('effects'))
                db.createObjectStore('effects', { keyPath: 'id' });
            if (!db.objectStoreNames.contains('native_values')) db.createObjectStore('native_values');
            if (!db.objectStoreNames.contains('child_runs'))
                db.createObjectStore('child_runs');
        };
        const db = await new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
        return new StudioStore(db);
    }
    constructor(db) { this.db = db; }
    async get(store, key) { return new Promise((resolve, reject) => { const tx = this.db.transaction(store); const r = tx.objectStore(store).get(key); r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); }); }
    async all(store) { return new Promise((resolve, reject) => { const r = this.db.transaction(store).objectStore(store).getAll(); r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); }); }
    async transaction(stores, write) { return new Promise((resolve, reject) => { const tx = this.db.transaction(stores, 'readwrite'); tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error ?? new Error('Storage transaction aborted')); try {
        write(tx);
    }
    catch (error) {
        tx.abort();
        reject(error);
    } }); }
    async put(store, key, value) { await this.transaction([store], tx => tx.objectStore(store).put(value, key)); }
    async effect(value) { await this.transaction(['effects'], tx => tx.objectStore('effects').put(value)); }
    async commit(app, record) {
        await this.transaction(['states', 'history'], tx => { tx.objectStore('states').put(record, app); tx.objectStore('history').add({ app, ...record }); });
    }
    async history(app) { return (await this.all('history')).filter(row => row.app === app); }
    close() { this.db.close(); }
}
