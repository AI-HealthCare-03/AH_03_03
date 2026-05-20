import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import {
  getMe,
  login as loginWithFastApi,
  signup as signupWithFastApi,
  type BackendUser,
  type SignupPayload,
} from "../api/auth";
import { clearStoredAccessToken, getStoredAccessToken, setStoredAccessToken } from "../api/client";

type AuthContextValue = {
  backendUser: BackendUser | null;
  loading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (payload: SignupPayload) => Promise<void>;
  logout: () => Promise<void>;
  refreshBackendUser: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [backendUser, setBackendUser] = useState<BackendUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshBackendUser = async () => {
    if (!getStoredAccessToken()) {
      setBackendUser(null);
      return;
    }
    setBackendUser(await getMe());
  };

  useEffect(() => {
    const restoreSession = async () => {
      try {
        await refreshBackendUser();
      } catch {
        clearStoredAccessToken();
        setBackendUser(null);
      } finally {
        setLoading(false);
      }
    };
    void restoreSession();
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      backendUser,
      loading,
      isAuthenticated: Boolean(backendUser && getStoredAccessToken()),
      login: async (email: string, password: string) => {
        const response = await loginWithFastApi({ email, password });
        setStoredAccessToken(response.access_token);
        await refreshBackendUser();
      },
      signup: async (payload: SignupPayload) => {
        await signupWithFastApi(payload);
        const response = await loginWithFastApi({ email: payload.email, password: payload.password });
        setStoredAccessToken(response.access_token);
        await refreshBackendUser();
      },
      logout: async () => {
        clearStoredAccessToken();
        setBackendUser(null);
      },
      refreshBackendUser,
    }),
    [backendUser, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return value;
}
