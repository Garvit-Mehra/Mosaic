import { useRef } from "react";
import { Plus } from "lucide-react";

interface FileUploadButtonProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function FileUploadButton({ onFileSelect, disabled = false }: FileUploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    onFileSelect(file);
    
    // Reset the input so the same file can be selected again if removed
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="relative group">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleUpload}
        className="hidden"
        accept=".pdf,.txt,.png,.jpg,.jpeg"
        disabled={disabled}
      />
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled}
        className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
          disabled ? 'opacity-50 cursor-not-allowed bg-[var(--hover)]' : 'bg-[var(--hover)] hover:bg-[var(--border)] active:scale-95'
        }`}
        title="Upload Document"
      >
        <Plus className="w-5 h-5 text-gray-400 group-hover:text-white transition-colors" />
      </button>
    </div>
  );
}
