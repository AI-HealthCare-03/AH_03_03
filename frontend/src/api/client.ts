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
  skipRefresh?: boolean;
  hasRetriedAfterRefresh?: boolean;
};

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
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

function shouldSkipRefresh(path: string, options: ApiOptions): boolean {
  if (options.skipAuth || options.skipRefresh || options.hasRetriedAfterRefresh) {
    return true;
  }

  return ["/auth/login", "/auth/signup", "/auth/token/refresh", "/auth/logout"].some((authPath) =>
    path.startsWith(authPath),
  );
}

async function parseErrorMessage(response: Response): Promise<string> {
  const fallbackMessage =
    response.status === 502 || response.status === 503 || response.status === 504
      ? "서버 연결이 일시적으로 불안정합니다. 잠시 후 다시 시도해주세요."
      : `API 요청 실패 (${response.status})`;
  try {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const errorBody = (await response.json()) as { detail?: unknown; message?: unknown; msg?: unknown };
      return formatApiErrorDetail(errorBody.detail ?? errorBody.message ?? errorBody.msg) ?? fallbackMessage;
    }

    const text = (await response.text()).trim();
    if (contentType.includes("text/html") || /^<!doctype html|^<html/i.test(text)) {
      return fallbackMessage;
    }
    return text || fallbackMessage;
  } catch {
    // 응답 본문이 깨져도 화면에는 상태 코드 기반 안내를 유지한다.
  }
  return fallbackMessage;
}

export async function refreshAccessToken(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/auth/token/refresh`, {
    method: "GET",
    credentials: "include",
  });

  if (!response.ok) {
    clearStoredAccessToken();
    throw new Error(await parseErrorMessage(response));
  }

  const data = (await response.json()) as { access_token?: string };
  if (!data.access_token) {
    clearStoredAccessToken();
    throw new Error("토큰 갱신 응답이 올바르지 않습니다.");
  }

  setStoredAccessToken(data.access_token);
  return data.access_token;
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
    credentials: options.credentials ?? "include",
    body:
      options.body === undefined || options.body === null || isFormData
        ? (options.body as BodyInit | null | undefined)
        : JSON.stringify(options.body),
  });

  if (response.status === 401 && !shouldSkipRefresh(path, options)) {
    try {
      await refreshAccessToken();
      return apiRequest<T>(path, { ...options, hasRetriedAfterRefresh: true });
    } catch {
      clearStoredAccessToken();
      throw new Error("로그인이 만료되었습니다. 다시 로그인해주세요.");
    }
  }

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function apiBlobRequest(path: string, options: ApiOptions = {}): Promise<Blob> {
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
    credentials: options.credentials ?? "include",
    body:
      options.body === undefined || options.body === null || isFormData
        ? (options.body as BodyInit | null | undefined)
        : JSON.stringify(options.body),
  });

  if (response.status === 401 && !shouldSkipRefresh(path, options)) {
    try {
      await refreshAccessToken();
      return apiBlobRequest(path, { ...options, hasRetriedAfterRefresh: true });
    } catch {
      clearStoredAccessToken();
      throw new Error("로그인이 만료되었습니다. 다시 로그인해주세요.");
    }
  }

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.blob();
}
