/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_STATIC?: string;
  readonly VITE_MOCK_CHAT?: string;
  readonly VITE_API_BASE?: string;
  readonly VITE_DATA_BASE?: string;
  readonly VITE_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
