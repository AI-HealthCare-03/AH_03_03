export type ApiValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | ApiValue[]
  | { [key: string]: ApiValue };

type ApiOptions = Omit<RequestInit, "body"> & {
  body?: BodyInit | Record<string, ApiValue> | null;
  skipAuth?: boolean;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
export const ACCESS_TOKEN_STORAGE_KEY = "ai_health_access_token";

export function getStoredAccessToken(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
}

export function setStoredAccessToken(token: string): void {
  window.localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
}

export function clearStoredAccessToken(): void {
  window.localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
}

function formatApiErrorDetail(detail: unknown): string | null {
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object") {
          const errorItem = item as { loc?: unknown; msg?: unknown; message?: unknown };
          const rawMessage = errorItem.msg ?? errorItem.message;
          if (rawMessage) {
            const message = String(rawMessage).replace(/^Value error,\s*/i, "");
            const field = Array.isArray(errorItem.loc)
              ? errorItem.loc
                  .filter((part) => part !== "body")
                  .map(String)
                  .join(".")
              : "";
            return field ? `${field}: ${message}` : message;
          }
        }
        return JSON.stringify(item);
      })
      .filter(Boolean);
    return messages.length > 0 ? messages.join("\n") : null;
  }

  if (detail && typeof detail === "object") {
    if ("msg" in detail) {
      return String((detail as { msg: unknown }).msg);
    }
    if ("message" in detail) {
      return String((detail as { message: unknown }).message);
    }
    return JSON.stringify(detail);
  }

  return null;
}

export async function apiRequest<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const isFormData = options.body instanceof FormData;

  if (!isFormData && options.body !== undefined && options.body !== null) {
    headers.set("Content-Type", "application/json");
  }

  if (!options.skipAuth) {
    const token = getStoredAccessToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    body:
      options.body === undefined || options.body === null || isFormData
        ? (options.body as BodyInit | null | undefined)
        : JSON.stringify(options.body),
  });

  if (!response.ok) {
    let message = `API 요청 실패 (${response.status})`;
    try {
      const errorBody = (await response.json()) as { detail?: unknown; message?: unknown };
      message = formatApiErrorDetail(errorBody.detail ?? errorBody.message) ?? message;
    } catch {
      // Keep the status based fallback message.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}
