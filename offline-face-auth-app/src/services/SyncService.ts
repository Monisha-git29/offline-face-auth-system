import BiometricService, { SyncItem, UserProfile } from './BiometricService';

export interface RestoreResponse {
  success: boolean;
  userId: string;
  name: string;
  encryptedEmbedding: string;
  salt: string;
  iv: string;
}

class SyncService {
  private awsGatewayUrl: string = "https://example-api-gateway.execute-api.us-east-1.amazonaws.com/prod";
  private isSyncing: boolean = false;

  setGatewayUrl(url: string) {
    this.awsGatewayUrl = url;
  }

  getGatewayUrl(): string {
    return this.awsGatewayUrl;
  }

  /**
   * Checks network connectivity to the AWS Gateway.
   */
  async checkConnectivity(): Promise<boolean> {
    try {
      const response = await fetch(`${this.awsGatewayUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        // Set short timeout to verify actual connectivity quickly
        signal: (AbortSignal as any).timeout(3000)
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  /**
   * Runs the local synchronization queue.
   * Uploads enrollments, logs, and deletions sequentially to prevent duplicates.
   */
  async runSyncQueue(): Promise<{ successCount: number; failedCount: number }> {
    if (this.isSyncing) return { successCount: 0, failedCount: 0 };
    this.isSyncing = true;

    let successCount = 0;
    let failedCount = 0;

    try {
      const queue: SyncItem[] = await BiometricService.getSyncQueue();
      if (queue.length === 0) {
        this.isSyncing = false;
        return { successCount, failedCount };
      }

      const isConnected = await this.checkConnectivity();
      if (!isConnected) {
        this.isSyncing = false;
        return { successCount, failedCount: queue.length };
      }

      for (const item of queue) {
        let endpoint = '';
        let method = 'POST';
        const bodyObj = JSON.parse(item.payload);

        switch (item.actionType) {
          case 'ENROLL':
            endpoint = '/sync/enroll';
            break;
          case 'LOG':
            endpoint = '/sync/log';
            break;
          case 'DELETE':
            endpoint = '/sync/delete';
            break;
        }

        try {
          const response = await fetch(`${this.awsGatewayUrl}${endpoint}`, {
            method,
            headers: {
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            body: JSON.stringify(bodyObj),
            signal: (AbortSignal as any).timeout(5000),
          });

          if (response.status === 200 || response.status === 201) {
            await BiometricService.deleteSyncItem(item.syncId);
            successCount++;
          } else {
            failedCount++;
            // Stop queue execution on error to preserve strict sequencing
            break;
          }
        } catch (err) {
          failedCount++;
          break; // Stop queue execution on connection loss
        }
      }
    } catch (e) {
      console.error("Sync queue failed to run", e);
    } finally {
      this.isSyncing = false;
    }

    return { successCount, failedCount };
  }

  /**
   * Downloads encrypted user details and loads them into local SQLite.
   * Authentication decryption occurs on first match attempt with correct recovery PIN.
   */
  async restoreUserData(userId: string): Promise<boolean> {
    try {
      const isConnected = await this.checkConnectivity();
      if (!isConnected) return false;

      const response = await fetch(`${this.awsGatewayUrl}/sync/restore?user_id=${encodeURIComponent(userId)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: (AbortSignal as any).timeout(6000),
      });

      if (response.status !== 200) return false;

      const data: RestoreResponse = await response.json();
      if (!data.userId || !data.encryptedEmbedding || !data.salt || !data.iv) return false;

      // Register the encrypted embedding details in local SQLite
      const success = await BiometricService.restoreUser(
        data.userId,
        data.name,
        data.encryptedEmbedding,
        data.salt,
        data.iv
      );

      return success;
    } catch (e) {
      console.error("Restore failed for user " + userId, e);
      return false;
    }
  }
}

export default new SyncService();
