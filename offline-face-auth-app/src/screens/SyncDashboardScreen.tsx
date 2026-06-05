import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert, SafeAreaView, TextInput, ScrollView, ActivityIndicator } from 'react-native';
import SyncService from '../services/SyncService';
import BiometricService from '../services/BiometricService';

export default function SyncDashboardScreen() {
  const [isConnected, setIsConnected] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const [lastSyncTime, setLastSyncTime] = useState<string>('Never');
  const [syncing, setSyncing] = useState(false);

  // Restore state variables
  const [restoreUserId, setRestoreUserId] = useState('');
  const [restoring, setRestoring] = useState(false);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 10000); // Check network/queue status every 10s
    return () => clearInterval(interval);
  }, []);

  const checkStatus = async () => {
    const connected = await SyncService.checkConnectivity();
    setIsConnected(connected);

    const queue = await BiometricService.getSyncQueue();
    setPendingCount(queue.length);
  };

  const handleSyncNow = async () => {
    if (syncing) return;
    setSyncing(true);

    try {
      const { successCount, failedCount } = await SyncService.runSyncQueue();
      setLastSyncTime(new Date().toLocaleTimeString());
      await checkStatus();

      if (successCount > 0) {
        Alert.alert("Sync Successful", `Uploaded ${successCount} records to the cloud database successfully.`);
      } else if (failedCount > 0) {
        Alert.alert("Sync Blocked", "Failed to connect to AWS Gateway. Retrying in background...");
      } else {
        Alert.alert("Up to Date", "Local biometric registry is already in sync with AWS cloud.");
      }
    } catch (e: any) {
      Alert.alert("Sync Error", e.message || "Failed to process synchronization pipeline.");
    } finally {
      setSyncing(false);
    }
  };

  const handleRestore = async () => {
    if (!restoreUserId.trim()) {
      return Alert.alert("Input Error", "Please provide the User ID you wish to restore.");
    }

    setRestoring(true);
    try {
      const success = await SyncService.restoreUserData(restoreUserId);
      if (success) {
        Alert.alert(
          "Restore Completed", 
          `Encrypted profile for ${restoreUserId} has been recovered successfully from the cloud database. Verify matching by authenticating using your face and PIN.`
        );
        setRestoreUserId('');
        checkStatus();
      } else {
        Alert.alert(
          "Restore Failed", 
          "Failed to restore profile. Verify network connectivity, or check if the User ID is enrolled in the cloud database."
        );
      }
    } catch (err: any) {
      Alert.alert("Restore Failed", err.message || "An error occurred during restore operations.");
    } finally {
      setRestoring(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.statusCard}>
          <Text style={styles.cardHeader}>Sync Status</Text>

          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Network Status</Text>
            <View style={[styles.statusIndicator, isConnected ? styles.online : styles.offline]} />
            <Text style={[styles.statusText, isConnected ? styles.onlineText : styles.offlineText]}>
              {isConnected ? 'ONLINE' : 'OFFLINE'}
            </Text>
          </View>

          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Pending Sync Records</Text>
            <Text style={styles.statusVal}>{pendingCount}</Text>
          </View>

          <View style={styles.statusRow}>
            <Text style={styles.statusLabel}>Last Synchronized</Text>
            <Text style={styles.statusVal}>{lastSyncTime}</Text>
          </View>

          {syncing ? (
            <ActivityIndicator size="small" color="#3B82F6" style={{ marginTop: 20 }} />
          ) : (
            <TouchableOpacity 
              style={[styles.button, { backgroundColor: isConnected ? '#EA580C' : '#334155' }]} 
              onPress={handleSyncNow}
            >
              <Text style={styles.buttonText}>Synchronize Now</Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={styles.restoreCard}>
          <Text style={styles.cardHeader}>Cloud Backup Restore</Text>
          <Text style={styles.cardInfo}>
            If you reinstalled the app or switched devices, download your encrypted credentials below.
          </Text>

          <TextInput
            style={styles.input}
            value={restoreUserId}
            onChangeText={setRestoreUserId}
            placeholder="Enter User ID (e.g. NHAI-101)"
            placeholderTextColor="#64748B"
            autoCapitalize="characters"
          />

          {restoring ? (
            <ActivityIndicator size="small" color="#3B82F6" style={{ marginTop: 16 }} />
          ) : (
            <TouchableOpacity style={styles.restoreBtn} onPress={handleRestore}>
              <Text style={styles.restoreBtnText}>Download & Restore</Text>
            </TouchableOpacity>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  container: {
    padding: 24,
    gap: 20,
  },
  statusCard: {
    backgroundColor: '#1E293B',
    borderRadius: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: '#334155',
  },
  cardHeader: {
    fontSize: 20,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 16,
    fontFamily: 'Outfit-Bold',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  statusLabel: {
    color: '#94A3B8',
    fontSize: 14,
    flex: 1,
    fontFamily: 'Outfit-Regular',
  },
  statusVal: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '600',
    fontFamily: 'Outfit-Medium',
  },
  statusIndicator: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 6,
  },
  statusText: {
    fontSize: 13,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
  online: {
    backgroundColor: '#10B981',
  },
  offline: {
    backgroundColor: '#EF4444',
  },
  onlineText: {
    color: '#10B981',
  },
  offlineText: {
    color: '#EF4444',
  },
  button: {
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 20,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
  restoreCard: {
    backgroundColor: '#1E293B',
    borderRadius: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: '#334155',
  },
  cardInfo: {
    fontSize: 13,
    color: '#94A3B8',
    marginBottom: 16,
    lineHeight: 18,
    fontFamily: 'Outfit-Regular',
  },
  input: {
    backgroundColor: '#0F172A',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: '#F8FAFC',
    borderWidth: 1,
    borderColor: '#334155',
    fontSize: 14,
    marginBottom: 16,
    fontFamily: 'Outfit-Regular',
  },
  restoreBtn: {
    backgroundColor: '#2563EB',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  restoreBtnText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
});
