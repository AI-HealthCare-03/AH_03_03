import { Navigate, Outlet } from "react-router-dom";

import { useAuth } from "./AuthContext";
import Loading from "../components/Loading";

export default function ProtectedRoute() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <Loading />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <Outlet />;
}
