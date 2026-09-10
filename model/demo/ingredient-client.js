export class IngredientBrowserClient {
  constructor() {
    this.worker = new Worker(new URL("./ingredient-worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId = 0;
    this.worker.addEventListener("message", ({ data }) => {
      const pending = this.pending.get(data.id);
      if (!pending) return;
      if (data.progress) {
        if (pending.progress) pending.progress(data.progress);
        return;
      }
      this.pending.delete(data.id);
      if (data.error) {
        const error = new Error(data.error.message);
        error.name = data.error.name;
        pending.reject(error);
      } else {
        pending.resolve(data.result);
      }
    });
    this.worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "The ingredient-search worker failed.");
      this.pending.forEach((request) => request.reject(error));
      this.pending.clear();
    });
  }

  request(operation, payload, progress = null) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, progress });
      this.worker.postMessage({ id, operation, ...payload });
    });
  }

  load(indexUrl, indexRecord, policy, progress) {
    return this.request("load", { index_url: indexUrl.href, index_record: indexRecord, policy }, progress);
  }

  search(query, ranking) {
    return this.request("search", { query, ranking });
  }

  close() {
    this.worker.terminate();
    this.pending.forEach((request) => request.reject(new Error("The ingredient index has been closed.")));
    this.pending.clear();
  }
}
