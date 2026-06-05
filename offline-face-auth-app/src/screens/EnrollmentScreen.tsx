import React, { useState, useRef } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, ActivityIndicator, Alert, SafeAreaView, ScrollView } from 'react-native';
import { Camera, useCameraDevice, useCameraPermission, CameraRef } from 'react-native-vision-camera';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import BiometricService from '../services/BiometricService';

type EnrollmentScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Enrollment'>;

interface Props {
  navigation: EnrollmentScreenNavigationProp;
}

export default function EnrollmentScreen({ navigation }: Props) {
  const [userId, setUserId] = useState('');
  const [name, setName] = useState('');
  const pin = '0000'; // Hardcoded internal PIN to bypass recovery PIN requirement in UI
  const [loading, setLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const { hasPermission, requestPermission } = useCameraPermission();

  const cameraRef = useRef<CameraRef>(null);
  const device = useCameraDevice('front');

  const handleEnroll = async () => {
    if (!userId.trim()) return Alert.alert("Input Error", "Please provide a valid User ID.");
    if (!name.trim()) return Alert.alert("Input Error", "Please provide the user's name.");

    if (!cameraActive) {
      if (!hasPermission) {
        const granted = await requestPermission();
        if (!granted) {
          Alert.alert(
            "Permission Denied",
            "Camera permission is required to capture face biometrics. Please enable it in Android Settings -> Apps -> FaceAuth -> Permissions."
          );
          return;
        }
      }
      setCameraActive(true);
      return;
    }

    if (!device) {
      return Alert.alert("Device Error", "No front camera found on this device.");
    }

    setLoading(true);
    try {
      if (!cameraRef.current) {
        throw new Error("Camera ref is not initialized.");
      }

      // 1. Capture silent snapshot
      const image = await cameraRef.current.takeSnapshot();
      if (!image) {
        throw new Error("Could not capture photo from hardware.");
      }

      const filePath = await image.saveToTemporaryFileAsync('jpg', 80);
      if (!filePath) {
        throw new Error("Could not save snapshot to temporary file.");
      }

      // 2. Convert file path to base64 using the fetch blob method
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

      // 3. Call enrollUser native module
      const res = await BiometricService.enrollUser(userId, name, base64Image, pin);
      if (res.success) {
        Alert.alert("Success", `User ${name} has been enrolled successfully.`, [
          { text: "OK", onPress: () => navigation.navigate('Home') }
        ]);
      } else {
        Alert.alert("Enrollment Failed", "Quality check failed. Please ensure neutral pose and good lighting.");
      }
    } catch (err: any) {
      console.error(err);
      Alert.alert("Enrollment Failed", err.message || "An unexpected error occurred during face registration.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        {!cameraActive ? (
          <View style={styles.formCard}>
            <Text style={styles.cardHeader}>Enrollment Credentials</Text>
            <Text style={styles.cardInfo}>
              Configure employee credentials to register face biometrics.
            </Text>

            <View style={styles.inputContainer}>
              <Text style={styles.label}>User ID (e.g., NHAI-101)</Text>
              <TextInput
                style={styles.input}
                value={userId}
                onChangeText={setUserId}
                placeholder="Enter unique ID"
                placeholderTextColor="#64748B"
                autoCapitalize="characters"
              />
            </View>

            <View style={styles.inputContainer}>
              <Text style={styles.label}>Full Name</Text>
              <TextInput
                style={styles.input}
                value={name}
                onChangeText={setName}
                placeholder="Enter full name"
                placeholderTextColor="#64748B"
              />
            </View>

            <TouchableOpacity style={styles.button} onPress={handleEnroll}>
              <Text style={styles.buttonText}>Activate Camera & Enroll</Text>
            </TouchableOpacity>
          </View>
        ) : (
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
              {/* Overlay guides */}
              <View style={styles.overlayFrame} />
            </View>

            <Text style={styles.guideText}>
              Center your face inside the frame, maintain neutral expression, and click Enroll.
            </Text>

            {loading ? (
              <ActivityIndicator size="large" color="#3B82F6" style={{ marginTop: 24 }} />
            ) : (
              <View style={styles.actionRow}>
                <TouchableOpacity
                  style={[styles.actionBtn, styles.cancelBtn]}
                  onPress={() => setCameraActive(false)}
                >
                  <Text style={styles.actionBtnText}>Cancel</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.actionBtn, styles.captureBtn]}
                  onPress={handleEnroll}
                >
                  <Text style={styles.actionBtnText}>Capture & Register</Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        )}
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
    justifyContent: 'center',
    alignItems: 'center',
  },
  formCard: {
    backgroundColor: '#1E293B',
    borderRadius: 20,
    padding: 24,
    width: '100%',
    borderWidth: 1,
    borderColor: '#334155',
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
    fontFamily: 'Outfit-Regular',
  },
  inputContainer: {
    marginBottom: 16,
    width: '100%',
  },
  label: {
    color: '#94A3B8',
    fontSize: 12,
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
    fontSize: 15,
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
    borderColor: '#3B82F6',
    backgroundColor: '#1E293B',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
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
    borderColor: 'rgba(255, 255, 255, 0.4)',
    borderStyle: 'dashed',
    top: 40,
  },
  guideText: {
    color: '#94A3B8',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 24,
    lineHeight: 18,
    fontFamily: 'Outfit-Regular',
  },
  actionRow: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 24,
    width: '100%',
  },
  actionBtn: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  cancelBtn: {
    backgroundColor: '#334155',
  },
  captureBtn: {
    backgroundColor: '#2563EB',
  },
  actionBtnText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '700',
    fontFamily: 'Outfit-Bold',
  },
});
