type Handler = (event: ProgressEvent) => void;

class MiniTarget {
  private handlers = new Map<string, Handler[]>();

  addEventListener(type: string, handler: Handler): void {
    this.handlers.set(type, [...(this.handlers.get(type) ?? []), handler]);
  }

  dispatch(type: string, init: { loaded?: number; total?: number; lengthComputable?: boolean } = {}): void {
    const event = { type, loaded: 0, total: 0, lengthComputable: false, ...init } as unknown as ProgressEvent;
    for (const handler of this.handlers.get(type) ?? []) handler(event);
  }
}

/** Minimal controllable XMLHttpRequest stand-in. Instances are recorded in `MockXhr.instances`. */
export class MockXhr extends MiniTarget {
  static instances: MockXhr[] = [];

  readonly upload = new MiniTarget();
  method = "";
  url = "";
  headers: Record<string, string> = {};
  body: unknown = null;
  status = 0;
  responseText = "";
  responseHeaders: Record<string, string> = {};
  aborted = false;

  constructor() {
    super();
    MockXhr.instances.push(this);
  }

  open(method: string, url: string): void {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string): void {
    this.headers[name] = value;
  }

  getResponseHeader(name: string): string | null {
    return this.responseHeaders[name] ?? null;
  }

  send(body: unknown): void {
    this.body = body;
  }

  abort(): void {
    this.aborted = true;
    this.dispatch("abort");
  }

  /** Test driver: emit an upload progress event. */
  progress(loaded: number, total: number): void {
    this.upload.dispatch("progress", { loaded, total, lengthComputable: true });
  }

  /** Test driver: finish the request with a status and JSON body. */
  respond(status: number, body: unknown, headers: Record<string, string> = {}): void {
    this.status = status;
    this.responseText = typeof body === "string" ? body : JSON.stringify(body);
    this.responseHeaders = headers;
    this.dispatch("load");
  }

  fail(): void {
    this.dispatch("error");
  }

  static reset(): void {
    MockXhr.instances = [];
  }

  static latest(): MockXhr {
    const xhr = MockXhr.instances.at(-1);
    if (!xhr) throw new Error("No XHR was created");
    return xhr;
  }
}
