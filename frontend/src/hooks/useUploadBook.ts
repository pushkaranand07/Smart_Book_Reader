import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { uploadBook } from "../api/client";

/** Upload mutation that also exposes real XHR upload progress (0..1). */
export function useUploadBook() {
  const [progress, setProgress] = useState(0);
  const mutation = useMutation({
    mutationFn: (file: File) => {
      setProgress(0);
      return uploadBook(file, { onProgress: setProgress });
    },
  });
  return { ...mutation, progress };
}
