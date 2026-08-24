import { useRef } from "react";
import { Plus } from "lucide-react";
import { motion } from "framer-motion";

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
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.9 }}
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled}
        className={`w-9 h-9 rounded-full flex items-center justify-center transition-all mosaic-glass-button ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[rgba(255,255,255,0.2)]'
        }`}
        title="Upload Document"
      >
        <Plus className="w-5 h-5 text-[var(--color1)] opacity-70 group-hover:opacity-100 transition-opacity" />
      </motion.button>
    </div>
  );
}
