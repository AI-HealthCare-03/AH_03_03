type ConfirmDialogProps = {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  showCancel?: boolean;
  showActions?: boolean;
  tone?: "default" | "danger";
  onCancel?: () => void;
  onConfirm: () => void;
};

export default function ConfirmDialog({
  title,
  message,
  confirmLabel = "확인",
  cancelLabel = "취소",
  showActions = true,
  showCancel = true,
  tone = "default",
  onCancel,
  onConfirm,
}: ConfirmDialogProps) {
  return (
    <div className="dialog-backdrop" role="presentation">
      <div aria-modal="true" className="confirm-dialog" role="dialog">
        <h2>{title}</h2>
        <p>{message}</p>
        {showActions && (
          <div className="button-row">
            {showCancel && (
              <button className="btn-secondary" onClick={onCancel} type="button">
                {cancelLabel}
              </button>
            )}
            <button className={tone === "danger" ? "btn-danger" : "btn-primary"} onClick={onConfirm} type="button">
              {confirmLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
