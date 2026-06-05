import { NativeModules } from 'react-native';

const { FaceAuthModule } = NativeModules;

export interface UserProfile {
  userId: string;
  name: string;
  createdAt: string;
}

export interface SyncItem {
  syncId: string;
  actionType: 'ENROLL' | 'LOG' | 'DELETE';
  payload: string;
  createdAt: string;
}

export interface AuthLog {
  logId: string;
  userId: string;
  similarityScore: number;
  livenessScore: number;
  timestamp: string;
  status: 'SUCCESS' | 'FAILED';
}

export interface LivenessResponse {
  success: boolean;
  currentChallenge: 'BLINK' | 'TURN_LEFT' | 'TURN_RIGHT' | 'SMILE' | 'PASSED';
  challengeProgress: boolean[];
  livenessPassed: boolean;
  error: 'FACE_NOT_DETECTED' | 'FACE_TOO_SMALL' | 'LANDMARKS_UNSTABLE' | null;
}

export interface AuthResponse {
  success: boolean;
  userId: string | null;
  similarityScore: number;
  name?: string;
}

class BiometricService {
  /**
   * Initializes the face authentication modules and local databases.
   * @param dbName Name of the database file (e.g. "face_auth.db")
   */
  async initialize(dbName: string = "face_auth.db"): Promise<boolean> {
    return FaceAuthModule.initialize(dbName);
  }

  /**
   * Resets active liveness session challenges and timers.
   */
  async resetLiveness(): Promise<boolean> {
    return FaceAuthModule.resetLiveness();
  }

  /**
   * Enrolls a new user offline.
   * @param userId Unique identifier (Employee ID, etc.)
   * @param name Full name of the user
   * @param imageBase64 Base64 representation of the camera crop
   * @param pin Passphrase/PIN for encryption derivation
   */
  async enrollUser(
    userId: string,
    name: string,
    imageBase64: string,
    pin: string
  ): Promise<{ success: boolean; userId: string }> {
    return FaceAuthModule.enrollUser(userId, name, imageBase64, pin);
  }

  /**
   * Decrypts database embeddings with input PIN and matches against query image.
   * @param imageBase64 Base64 representation of the camera frame
   * @param pin Passphrase/PIN to decrypt embeddings
   */
  async authenticate(imageBase64: string, pin: string): Promise<AuthResponse> {
    return FaceAuthModule.authenticate(imageBase64, pin);
  }

  /**
   * Processes a video camera frame for active liveness checks.
   * @param imageBase64 Base64 representation of the frame
   * @param timestampMs Monotonic timestamp in milliseconds
   */
  async processFrame(imageBase64: string, timestampMs: number): Promise<LivenessResponse> {
    return FaceAuthModule.processFrame(imageBase64, timestampMs);
  }

  /**
   * Retrieves list of enrolled users.
   */
  async getUsers(): Promise<UserProfile[]> {
    return FaceAuthModule.getUsers();
  }

  /**
   * Removes user records locally.
   */
  async deleteUser(userId: string): Promise<boolean> {
    return FaceAuthModule.deleteUser(userId);
  }

  /**
   * Gets pending synchronization queue.
   */
  async getSyncQueue(): Promise<SyncItem[]> {
    return FaceAuthModule.getSyncQueue();
  }

  /**
   * Removes sync item after completion.
   */
  async deleteSyncItem(syncId: string): Promise<boolean> {
    return FaceAuthModule.deleteSyncItem(syncId);
  }

  /**
   * Restores a user profile from the cloud.
   */
  async restoreUser(
    userId: string,
    name: string,
    encryptedEmbedding: string,
    salt: string,
    iv: string
  ): Promise<boolean> {
    return FaceAuthModule.restoreUser(userId, name, encryptedEmbedding, salt, iv);
  }

  /**
   * Gets authentication logs locally.
   */
  async getAuthLogs(): Promise<AuthLog[]> {
    return FaceAuthModule.getAuthLogs();
  }
}

export default new BiometricService();
