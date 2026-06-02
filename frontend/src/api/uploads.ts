import { apiBlobRequest } from "./client";

export async function normalizeImageForPreview(file: File): Promise<Blob> {
  const formData = new FormData();
  formData.append("file", file);
  return apiBlobRequest("/uploads/normalize-image", {
    method: "POST",
    body: formData,
  });
}
