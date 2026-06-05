import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ActivityIndicator, Alert, SafeAreaView } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, CameraRef } from 'react-native-vision-camera';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import BiometricService, { LivenessResponse } from '../services/BiometricService';

type AuthenticationScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Authentication'>;

interface Props {
  navigation: AuthenticationScreenNavigationProp;
}

export default function AuthenticationScreen({ navigation }: Props) {
  const pin = '0000'; // Hardcoded internal PIN to bypass recovery PIN requirement
  const [cameraActive, setCameraActive] = useState(false);
  const { hasPermission, requestPermission } = useCameraPermission();
  
  // Liveness States
  const [currentChallenge, setCurrentChallenge] = useState<'BLINK' | 'TURN_LEFT' | 'TURN_RIGHT' | 'SMILE' | 'PASSED'>('BLINK');
  const [progress, setProgress] = useState<boolean[]>([false, false, false, false]);
  const [errorPrompt, setErrorPrompt] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);

  const cameraRef = useRef<CameraRef>(null);
  const device = useCameraDevice('front');
  const timerRef = useRef<any>(null);

  useEffect(() => {
    const initPermission = async () => {
      if (hasPermission) {
        setCameraActive(true);
        await BiometricService.resetLiveness();
      } else {
        const granted = await requestPermission();
        if (granted) {
          setCameraActive(true);
          await BiometricService.resetLiveness();
        } else {
          Alert.alert(
            "Permission Denied",
            "Camera access is required for biometric authentication. Please enable it in Android Settings -> Apps -> FaceAuth -> Permissions.",
            [
              { text: "Go Back", onPress: () => navigation.navigate('Home') }
            ]
          );
        }
      }
    };
    initPermission();
  }, []);

  useEffect(() => {
    if (cameraActive && hasPermission) {
      // Start frame capture loop
      timerRef.current = setInterval(captureAndProcessFrame, 150);
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [cameraActive, hasPermission, currentChallenge]);

  const captureAndProcessFrame = async () => {
    if (processing || !cameraRef.current) return;
    setProcessing(true);

    const tStart = Date.now();
    try {
      // 1. Take a silent snapshot of preview
      const tSnapStart = Date.now();
      const image = await cameraRef.current.takeSnapshot();
      const tSnapEnd = Date.now();
      if (!image) {
        setProcessing(false);
        return;
      }

      const tSaveStart = Date.now();
      const filePath = await image.saveToTemporaryFileAsync('jpg', 80);
      const tSaveEnd = Date.now();
      if (!filePath) {
        setProcessing(false);
        return;
      }

      // 2. Convert to Base64
      const tFetchStart = Date.now();
      const photoUri = `file://${filePath}`;
      const response = await fetch(photoUri);
      const blob = await response.blob();
      const base64Image = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const res = reader.result as string;
          resolve(res.split(',')[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
      const tFetchEnd = Date.now();

      const now = Date.now();

      // 3. Process frame through native active liveness module
      const tNativeStart = Date.now();
      const livenessRes: LivenessResponse = await BiometricService.processFrame(base64Image, now);
      const tNativeEnd = Date.now();

      console.log(`[PROFILE] Snap: ${tSnapEnd - tSnapStart}ms | Save: ${tSaveEnd - tSaveStart}ms | Base64: ${tFetchEnd - tFetchStart}ms | Native: ${tNativeEnd - tNativeStart}ms | Total: ${Date.now() - tStart}ms`);
      
      if (livenessRes.error) {
        // Map native error codes to friendly UI messages
        switch (livenessRes.error) {
          case 'FACE_NOT_DETECTED':
            setErrorPrompt("Place your face inside the frame");
            break;
          case 'FACE_TOO_SMALL':
            setErrorPrompt("Move closer to the camera");
            break;
          case 'LANDMARKS_UNSTABLE':
            setErrorPrompt("Hold still and look straight");
            break;
          default:
            setErrorPrompt("Hold still...");
        }
      } else {
        setErrorPrompt(null);
      }

      // Update liveness states
      setCurrentChallenge(livenessRes.currentChallenge);
      setProgress(livenessRes.challengeProgress);

      if (livenessRes.livenessPassed) {
        // Stop the frame timer immediately
        if (timerRef.current) clearInterval(timerRef.current);
        
        setErrorPrompt("Matching identity...");
        
        try {
          // 4. Run face recognition matching against SQLite registry using decrypted embeddings
          const authRes = await BiometricService.authenticate(base64Image, pin);
          
          setCameraActive(false);
          navigation.navigate('VerificationResult', {
            success: authRes.success,
            userId: authRes.userId,
            similarityScore: authRes.similarityScore,
            name: authRes.name
          });
        } catch (authErr) {
          console.error("Authentication match error:", authErr);
          setCameraActive(false);
          navigation.navigate('VerificationResult', {
            success: false,
            userId: 'UNKNOWN',
            similarityScore: 0,
            name: 'UNKNOWN'
          });
        }
      }
    } catch (err) {
      console.error("Frame processing error:", err);
    } finally {
      setProcessing(false);
    }
  };

  const getChallengeInstruction = () => {
    switch (currentChallenge) {
      case 'BLINK':
        return 'Blink your eyes';
      case 'TURN_LEFT':
        return 'Turn head to the left and return to center';
      case 'TURN_RIGHT':
        return 'Turn head to the right and return to center';
      case 'SMILE':
        return 'Smile for the camera';
      case 'PASSED':
        return 'Liveness Checks Passed!';
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.cameraWrapper}>
        <View style={styles.cameraContainer}>
          {device && hasPermission ? (
            <Camera
              ref={cameraRef}
              style={StyleSheet.absoluteFill}
              device={device}
              isActive={cameraActive}
              implementationMode="compatible"
            />
          ) : (
            <View style={styles.cameraError}>
              <Text style={styles.cameraErrorText}>Starting camera hardware...</Text>
            </View>
          )}
          {/* Custom oval frame overlay */}
          <View style={styles.overlayFrame} />
        </View>

        <View style={styles.challengeBox}>
          <Text style={styles.challengePrompt}>Liveness Challenge:</Text>
          <Text style={styles.challengeInstruction}>{getChallengeInstruction()}</Text>

          {errorPrompt && (
            <Text style={styles.errorText}>{errorPrompt}</Text>
          )}

          {/* Render checklist badges */}
          <View style={styles.progressRow}>
            {challenges.map((c, idx) => (
              <View
                key={c}
                style={[
                  styles.progressBadge,
                  progress[idx] ? styles.badgeChecked : styles.badgeUnchecked
                ]}
              >
                <Text style={styles.progressBadgeText}>{c}</Text>
              </View>
            ))}
          </View>
        </View>

        <TouchableOpacity
          style={styles.cancelBtn}
          onPress={() => {
            setCameraActive(false);
            navigation.navigate('Home');
          }}
        >
          <Text style={styles.cancelBtnText}>Exit</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const challenges = ["Blink", "Smile", "Left", "Right"];

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#070A13',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  pinCard: {
    backgroundColor: 'rgba(30, 41, 59, 0.45)',
    borderRadius: 20,
    padding: 24,
    width: '100%',
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.04)',
    alignItems: 'center',
  },
  cardHeader: {
    fontSize: 22,
    fontWeight: '700',
    color: '#F8FAFC',
    marginBottom: 8,
    fontFamily: 'Outfit-Bold',
  },
  cardInfo: {
    fontSize: 13,
    color: '#94A3B8',
    marginBottom: 24,
    lineHeight: 18,
    textAlign: 'center',
    fontFamily: 'Outfit-Regular',
  },
  pinInput: {
    backgroundColor: '#070A13',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: '#F8FAFC',
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    fontSize: 24,
    textAlign: 'center',
    letterSpacing: 8,
    width: '100%',
    marginBottom: 20,
    fontFamily: 'Outfit-Bold',
  },
  button: {
    backgroundColor: '#10B981',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    width: '100%',
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
  cameraWrapper: {
    width: '100%',
    alignItems: 'center',
  },
  cameraContainer: {
    width: 320,
    height: 320,
    borderRadius: 160,
    overflow: 'hidden',
    borderWidth: 4,
    borderColor: '#10B981',
    backgroundColor: '#070A13',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
    shadowColor: '#10B981',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 15,
    elevation: 10,
  },
  cameraError: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraErrorText: {
    color: '#94A3B8',
    fontSize: 14,
    fontFamily: 'Outfit-Regular',
  },
  overlayFrame: {
    position: 'absolute',
    width: 200,
    height: 240,
    borderRadius: 100,
    borderWidth: 2,
    borderColor: 'rgba(6, 182, 212, 0.5)',
    borderStyle: 'dashed',
    top: 40,
  },
  challengeBox: {
    backgroundColor: 'rgba(30, 41, 59, 0.45)',
    borderRadius: 20,
    padding: 20,
    width: '100%',
    marginTop: 24,
    borderWidth: 1.5,
    borderColor: 'rgba(255, 255, 255, 0.04)',
    alignItems: 'center',
  },
  challengePrompt: {
    color: '#06B6D4',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    fontFamily: 'Outfit-Bold',
  },
  challengeInstruction: {
    color: '#F8FAFC',
    fontSize: 18,
    fontWeight: '700',
    marginTop: 6,
    textAlign: 'center',
    minHeight: 48,
    fontFamily: 'Outfit-Bold',
  },
  errorText: {
    color: '#EF4444',
    fontSize: 13,
    fontWeight: '600',
    marginTop: 4,
    textAlign: 'center',
    fontFamily: 'Outfit-Medium',
  },
  progressRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 16,
  },
  progressBadge: {
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 8,
    borderWidth: 1,
  },
  badgeChecked: {
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    borderColor: '#10B981',
  },
  badgeUnchecked: {
    backgroundColor: 'transparent',
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  progressBadgeText: {
    color: '#F8FAFC',
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
    fontFamily: 'Outfit-Bold',
  },
  cancelBtn: {
    marginTop: 24,
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  cancelBtnText: {
    color: '#94A3B8',
    fontSize: 15,
    fontWeight: '600',
    fontFamily: 'Outfit-Medium',
  },
});
