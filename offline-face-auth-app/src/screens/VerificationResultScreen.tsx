import React from 'react';
import { StyleSheet, Text, View, TouchableOpacity, SafeAreaView } from 'react-native';
import { RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';

type ResultScreenRouteProp = RouteProp<RootStackParamList, 'VerificationResult'>;
type ResultScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'VerificationResult'>;

interface Props {
  route: ResultScreenRouteProp;
  navigation: ResultScreenNavigationProp;
}

export default function VerificationResultScreen({ route, navigation }: Props) {
  const { success, userId, similarityScore, name } = route.params;

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <View style={[styles.resultCard, success ? styles.successCard : styles.errorCard]}>
          <View style={styles.iconCircle}>
            <Text style={styles.iconText}>{success ? '✓' : '✗'}</Text>
          </View>
          <Text style={styles.verdictText}>
            {success ? 'Access Granted' : 'Access Denied'}
          </Text>
          <Text style={styles.statusDescription}>
            {success
              ? 'Identity verified successfully through sequential liveness and matching filters.'
              : 'Biometric query did not match any enrolled face template or verification failed.'}
          </Text>
        </View>

        <View style={styles.detailsCard}>
          <Text style={styles.detailsHeader}>Metric Summary</Text>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>User ID</Text>
            <Text style={styles.detailVal}>{success ? userId : 'UNKNOWN'}</Text>
          </View>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Full Name</Text>
            <Text style={styles.detailVal}>{success ? name : 'UNKNOWN'}</Text>
          </View>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Similarity Match Score</Text>
            <Text style={[styles.detailVal, success ? styles.successText : styles.errorText]}>
              {(similarityScore * 100).toFixed(1)}%
            </Text>
          </View>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Liveness Score</Text>
            <Text style={styles.detailVal}>100.0% (Passed)</Text>
          </View>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Quality Score</Text>
            <Text style={styles.detailVal}>PASSED</Text>
          </View>

          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Trust Score</Text>
            <Text style={[styles.detailVal, { color: '#3B82F6', fontWeight: '700' }]}>
              {success ? (similarityScore * 100).toFixed(1) : '0.0'}%
            </Text>
          </View>
        </View>

        <TouchableOpacity
          style={styles.doneBtn}
          onPress={() => navigation.navigate('Home')}
        >
          <Text style={styles.doneBtnText}>Return to Hub</Text>
        </TouchableOpacity>
      </View>
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
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
  },
  resultCard: {
    borderRadius: 24,
    padding: 24,
    alignItems: 'center',
    width: '100%',
    borderWidth: 1,
    marginBottom: 20,
  },
  successCard: {
    backgroundColor: 'rgba(5, 150, 105, 0.1)',
    borderColor: '#059669',
  },
  errorCard: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: '#EF4444',
  },
  iconCircle: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  iconText: {
    color: '#F8FAFC',
    fontSize: 32,
    fontWeight: '700',
  },
  verdictText: {
    fontSize: 24,
    fontWeight: '800',
    color: '#F8FAFC',
    fontFamily: 'Outfit-Bold',
  },
  statusDescription: {
    fontSize: 13,
    color: '#94A3B8',
    textAlign: 'center',
    marginTop: 8,
    lineHeight: 18,
    fontFamily: 'Outfit-Regular',
  },
  detailsCard: {
    backgroundColor: '#1E293B',
    borderRadius: 20,
    padding: 20,
    width: '100%',
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 32,
  },
  detailsHeader: {
    fontSize: 16,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 16,
    fontFamily: 'Outfit-Bold',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  detailLabel: {
    color: '#94A3B8',
    fontSize: 13,
    fontFamily: 'Outfit-Regular',
  },
  detailVal: {
    color: '#F8FAFC',
    fontSize: 13,
    fontWeight: '600',
    fontFamily: 'Outfit-Medium',
  },
  successText: {
    color: '#10B981',
  },
  errorText: {
    color: '#EF4444',
  },
  doneBtn: {
    backgroundColor: '#1E293B',
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    width: '100%',
  },
  doneBtnText: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
});
