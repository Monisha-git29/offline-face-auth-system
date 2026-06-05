import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, Alert, SafeAreaView, ScrollView, FlatList } from 'react-native';
import SyncService from '../services/SyncService';
import BiometricService, { AuthLog } from '../services/BiometricService';

export default function SettingsScreen() {
  const [gatewayUrl, setGatewayUrl] = useState('');
  const [similarityThreshold, setSimilarityThreshold] = useState('60'); // 0.60 in percentage
  const [logs, setLogs] = useState<AuthLog[]>([]);

  useEffect(() => {
    setGatewayUrl(SyncService.getGatewayUrl());
    fetchLogs();
  }, []);

  const fetchLogs = async () => {
    try {
      const authLogs = await BiometricService.getAuthLogs();
      setLogs(authLogs);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSaveSettings = () => {
    if (!gatewayUrl.trim()) return Alert.alert("Input Error", "Please provide a valid URL.");
    
    const threshVal = parseFloat(similarityThreshold);
    if (isNaN(threshVal) || threshVal < 0 || threshVal > 100) {
      return Alert.alert("Input Error", "Similarity Threshold must be between 0% and 100%.");
    }

    SyncService.setGatewayUrl(gatewayUrl.trim());
    Alert.alert("Success", "Settings saved successfully.");
  };

  const handleWipeDb = () => {
    Alert.alert(
      "CAUTION: Database Wipe",
      "Are you absolutely sure you want to wipe all local enrollments and logs? This will reset all biometric data on this device.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Wipe Data",
          style: "destructive",
          onPress: async () => {
            try {
              // Delete each user sequentially to populate sync queue, or wipe database
              const userList = await BiometricService.getUsers();
              for (const u of userList) {
                await BiometricService.deleteUser(u.userId);
              }
              // Wipe queue
              const queue = await BiometricService.getSyncQueue();
              for (const q of queue) {
                await BiometricService.deleteSyncItem(q.syncId);
              }
              Alert.alert("Success", "Local database wiped successfully.");
              fetchLogs();
            } catch (err: any) {
              Alert.alert("Wipe Failed", err.message || "Failed to wipe database.");
            }
          }
        }
      ]
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.card}>
          <Text style={styles.cardHeader}>Server Configuration</Text>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>AWS API Gateway URL</Text>
            <TextInput
              style={styles.input}
              value={gatewayUrl}
              onChangeText={setGatewayUrl}
              placeholder="https://example-api-gateway.execute-api..."
              placeholderTextColor="#64748B"
            />
          </View>

          <View style={styles.inputContainer}>
            <Text style={styles.label}>Cosine Match Threshold (%)</Text>
            <TextInput
              style={styles.input}
              value={similarityThreshold}
              onChangeText={setSimilarityThreshold}
              placeholder="60"
              placeholderTextColor="#64748B"
              keyboardType="numeric"
            />
          </View>

          <TouchableOpacity style={styles.button} onPress={handleSaveSettings}>
            <Text style={styles.buttonText}>Save Configurations</Text>
          </TouchableOpacity>
        </View>

        <View style={[styles.card, { borderColor: 'rgba(239, 68, 68, 0.4)' }]}>
          <Text style={[styles.cardHeader, { color: '#EF4444' }]}>Danger Zone</Text>
          <Text style={styles.cardInfo}>
            Wiping the local database clears all profiles and biometric templates. 
          </Text>
          <TouchableOpacity style={styles.wipeBtn} onPress={handleWipeDb}>
            <Text style={styles.wipeBtnText}>Wipe Database</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardHeader}>Audit Trail (Auth Logs)</Text>
          {logs.length === 0 ? (
            <Text style={styles.emptyLogsText}>No authentication logs found on this device.</Text>
          ) : (
            <View style={styles.logList}>
              {logs.slice(0, 10).map((log) => (
                <View key={log.logId} style={styles.logItem}>
                  <View style={styles.logMeta}>
                    <Text style={styles.logUser}>{log.userId}</Text>
                    <Text style={styles.logTime}>
                      {new Date(log.timestamp).toLocaleString()}
                    </Text>
                  </View>
                  <View style={styles.logResult}>
                    <Text style={log.status === 'SUCCESS' ? styles.logSuccess : styles.logFail}>
                      {log.status}
                    </Text>
                    <Text style={styles.logScore}>
                      Match: {(log.similarityScore * 100).toFixed(0)}%
                    </Text>
                  </View>
                </View>
              ))}
            </View>
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
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: '#334155',
  },
  cardHeader: {
    fontSize: 18,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 16,
    fontFamily: 'Outfit-Bold',
  },
  cardInfo: {
    fontSize: 13,
    color: '#94A3B8',
    marginBottom: 16,
    lineHeight: 18,
    fontFamily: 'Outfit-Regular',
  },
  inputContainer: {
    marginBottom: 16,
  },
  label: {
    color: '#94A3B8',
    fontSize: 11,
    fontWeight: '600',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    fontFamily: 'Outfit-Bold',
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
    fontFamily: 'Outfit-Regular',
  },
  button: {
    backgroundColor: '#2563EB',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 12,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
  wipeBtn: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: '#EF4444',
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  wipeBtnText: {
    color: '#EF4444',
    fontSize: 15,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
  emptyLogsText: {
    color: '#64748B',
    fontSize: 13,
    fontFamily: 'Outfit-Regular',
  },
  logList: {
    gap: 10,
  },
  logItem: {
    backgroundColor: '#0F172A',
    borderRadius: 12,
    padding: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#334155',
  },
  logMeta: {
    flex: 1,
  },
  logUser: {
    color: '#F8FAFC',
    fontWeight: '700',
    fontSize: 14,
    fontFamily: 'Outfit-Bold',
  },
  logTime: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 4,
    fontFamily: 'Outfit-Regular',
  },
  logResult: {
    alignItems: 'flex-end',
  },
  logSuccess: {
    color: '#10B981',
    fontWeight: '700',
    fontSize: 12,
    fontFamily: 'Outfit-Bold',
  },
  logFail: {
    color: '#EF4444',
    fontWeight: '700',
    fontSize: 12,
    fontFamily: 'Outfit-Bold',
  },
  logScore: {
    color: '#94A3B8',
    fontSize: 11,
    marginTop: 4,
    fontFamily: 'Outfit-Regular',
  },
});
