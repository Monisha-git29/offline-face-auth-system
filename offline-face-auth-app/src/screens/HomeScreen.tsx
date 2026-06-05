import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, SafeAreaView, ActivityIndicator } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import BiometricService from '../services/BiometricService';

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

interface Props {
  navigation: HomeScreenNavigationProp;
}

export default function HomeScreen({ navigation }: Props) {
  const [init, setInit] = useState(false);
  const [loading, setLoading] = useState(true);
  const [syncQueueCount, setSyncQueueCount] = useState(0);

  useEffect(() => {
    const initializeSDK = async () => {
      try {
        await BiometricService.initialize();
        setInit(true);
        const queue = await BiometricService.getSyncQueue();
        setSyncQueueCount(queue.length);
      } catch (err) {
        console.error("SDK Initialization failed", err);
      } finally {
        setLoading(false);
      }
    };

    const unsubscribe = navigation.addListener('focus', () => {
      BiometricService.getSyncQueue().then(queue => setSyncQueueCount(queue.length)).catch(() => {});
    });

    initializeSDK();
    return unsubscribe;
  }, [navigation]);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#06B6D4" />
        <Text style={styles.loadingText}>Loading Security Core...</Text>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <View style={styles.titleRow}>
            <Text style={styles.titleMain}>SafeGate</Text>
            <Text style={styles.titleSub}> Face Auth</Text>
          </View>
          <Text style={styles.tagline}>Autonomous Offline Biometric Entry</Text>
          <View style={styles.shieldBadge}>
            <Text style={styles.shieldBadgeText}>🛡️ SECURE LOCAL SHIELD ACTIVE</Text>
          </View>
        </View>

        <View style={styles.menuContainer}>
          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('Enrollment')}
          >
            <View style={[styles.iconBox, { backgroundColor: 'rgba(59, 130, 246, 0.15)', borderColor: 'rgba(59, 130, 246, 0.3)' }]}>
              <Text style={styles.iconGlyph}>👤</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>User Enrollment</Text>
              <Text style={styles.cardDesc}>Register biometric profiles and securely generate local encrypted templates.</Text>
            </View>
            <Text style={styles.cardArrow}>→</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('Authentication')}
          >
            <View style={[styles.iconBox, { backgroundColor: 'rgba(16, 185, 129, 0.15)', borderColor: 'rgba(16, 185, 129, 0.3)' }]}>
              <Text style={styles.iconGlyph}>🔒</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>Authenticate User</Text>
              <Text style={styles.cardDesc}>Execute instant local liveness checking and verify identity in real-time.</Text>
            </View>
            <Text style={styles.cardArrow}>→</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('RegisteredUsers')}
          >
            <View style={[styles.iconBox, { backgroundColor: 'rgba(139, 92, 246, 0.15)', borderColor: 'rgba(139, 92, 246, 0.3)' }]}>
              <Text style={styles.iconGlyph}>👥</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>Registered Users</Text>
              <Text style={styles.cardDesc}>Manage local templates database, review authorization logs, or delete profiles.</Text>
            </View>
            <Text style={styles.cardArrow}>→</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('SyncDashboard')}
          >
            <View style={[styles.iconBox, { backgroundColor: 'rgba(249, 115, 22, 0.15)', borderColor: 'rgba(249, 115, 22, 0.3)' }]}>
              <Text style={styles.iconGlyph}>🔄</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>Sync Dashboard</Text>
              <Text style={styles.cardDesc}>
                {syncQueueCount > 0
                  ? `${syncQueueCount} pending profile updates in queue.`
                  : 'All biometric templates and audit trails synchronized.'}
              </Text>
            </View>
            {syncQueueCount > 0 && (
              <View style={styles.countBadge}>
                <Text style={styles.countBadgeText}>{syncQueueCount}</Text>
              </View>
            )}
            <Text style={styles.cardArrow}>→</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.card}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('Settings')}
          >
            <View style={[styles.iconBox, { backgroundColor: 'rgba(100, 116, 139, 0.15)', borderColor: 'rgba(100, 116, 139, 0.3)' }]}>
              <Text style={styles.iconGlyph}>⚙️</Text>
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>App Settings</Text>
              <Text style={styles.cardDesc}>Calibrate match threshold, adjust liveness variables, and setup endpoints.</Text>
            </View>
            <Text style={styles.cardArrow}>→</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.footerText}>NHAI Secure Entry Core • OpenCV & TFLite Module</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#070A13',
  },
  container: {
    padding: 24,
    alignItems: 'center',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#070A13',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#06B6D4',
    marginTop: 16,
    fontSize: 16,
    fontFamily: 'Outfit-Medium',
  },
  header: {
    alignItems: 'center',
    marginBottom: 36,
    marginTop: 24,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  titleMain: {
    fontSize: 34,
    fontWeight: '900',
    color: '#06B6D4',
    fontFamily: 'Outfit-Bold',
    textShadowColor: 'rgba(6, 182, 212, 0.35)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 10,
  },
  titleSub: {
    fontSize: 34,
    fontWeight: '900',
    color: '#10B981',
    fontFamily: 'Outfit-Bold',
    textShadowColor: 'rgba(16, 185, 129, 0.35)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 10,
  },
  tagline: {
    fontSize: 13,
    color: '#64748B',
    marginTop: 6,
    letterSpacing: 0.5,
    fontFamily: 'Outfit-Regular',
  },
  shieldBadge: {
    backgroundColor: 'rgba(16, 185, 129, 0.08)',
    borderRadius: 20,
    paddingVertical: 6,
    paddingHorizontal: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: 'rgba(16, 185, 129, 0.25)',
  },
  shieldBadgeText: {
    color: '#10B981',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
    fontFamily: 'Outfit-Bold',
  },
  menuContainer: {
    width: '100%',
    gap: 16,
  },
  card: {
    backgroundColor: 'rgba(30, 41, 59, 0.45)',
    borderRadius: 20,
    padding: 18,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.04)',
    position: 'relative',
  },
  iconBox: {
    width: 48,
    height: 48,
    borderRadius: 14,
    borderWidth: 1.5,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  iconGlyph: {
    fontSize: 20,
  },
  cardContent: {
    flex: 1,
    paddingRight: 12,
  },
  cardTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#F8FAFC',
    fontFamily: 'Outfit-Bold',
  },
  cardDesc: {
    fontSize: 12,
    color: '#94A3B8',
    marginTop: 4,
    lineHeight: 16,
    fontFamily: 'Outfit-Regular',
  },
  cardArrow: {
    color: '#475569',
    fontSize: 18,
    fontWeight: 'bold',
  },
  countBadge: {
    position: 'absolute',
    top: 14,
    right: 36,
    backgroundColor: '#EF4444',
    borderRadius: 10,
    width: 18,
    height: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  countBadgeText: {
    color: '#FFFFFF',
    fontSize: 9,
    fontWeight: '800',
    fontFamily: 'Outfit-Bold',
  },
  footerText: {
    color: '#334155',
    fontSize: 10,
    marginTop: 48,
    marginBottom: 16,
    fontFamily: 'Outfit-Regular',
    letterSpacing: 0.2,
  },
});
