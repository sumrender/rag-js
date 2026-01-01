import multer from "multer";

const storage = multer.memoryStorage();

const fileFilter: multer.Options["fileFilter"] = (_req, file, cb) => {
  if (file.mimetype !== "text/plain" && file.mimetype !== "application/pdf") {
    cb(new Error("Only TXT and PDF files are allowed"));
    return;
  }
  cb(null, true);
};

export const uploadTxt = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10 MB
  },
});