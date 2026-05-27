export default function ErrorMessage({ message }: { message: string }) {
  return <div className="error-box">{message}</div>;
}
