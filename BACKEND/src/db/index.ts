import mongoose from 'mongoose';

export const connectDB = async (): Promise<void> => {
  try {
    const mongoUri: string = process.env.MONGO_URI ?? 'mongodb://localhost:27017/simple-rag';
    await mongoose.connect(mongoUri);
    console.log('MongoDB connected successfully');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
};

export const checkDbConnection = async (): Promise<boolean> => {
  try {
    if (mongoose.connection.readyState !== mongoose.ConnectionStates.connected) {
      return false;
    }
    if (mongoose.connection.db) {
      await mongoose.connection.db.admin().ping();
      return true;
    }
    return false;
  } catch {
    return false;
  }
};
