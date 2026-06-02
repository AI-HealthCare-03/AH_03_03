export function isHeicFile(file: File | null): boolean {
  if (!file) {
    return false;
  }

  const name = file.name.toLowerCase();
  const type = file.type.toLowerCase();
  return type === "image/heic" || type === "image/heif" || name.endsWith(".heic") || name.endsWith(".heif");
}
