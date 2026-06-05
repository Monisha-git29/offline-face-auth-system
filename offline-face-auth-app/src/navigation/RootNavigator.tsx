import React from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from '../screens/HomeScreen';
import EnrollmentScreen from '../screens/EnrollmentScreen';
import AuthenticationScreen from '../screens/AuthenticationScreen';
import VerificationResultScreen from '../screens/VerificationResultScreen';
import RegisteredUsersScreen from '../screens/RegisteredUsersScreen';
import SyncDashboardScreen from '../screens/SyncDashboardScreen';
import SettingsScreen from '../screens/SettingsScreen';

export type RootStackParamList = {
  Home: undefined;
  Enrollment: undefined;
  Authentication: undefined;
  VerificationResult: {
    success: boolean;
    userId: string | null;
    similarityScore: number;
    name?: string;
  };
  RegisteredUsers: undefined;
  SyncDashboard: undefined;
  Settings: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function RootNavigator() {
  return (
    <Stack.Navigator
      initialRouteName="Home"
      screenOptions={{
        headerStyle: {
          backgroundColor: '#1E293B', // Slate 800 (sleek dark palette)
        },
        headerTintColor: '#F8FAFC', // Slate 50
        headerTitleStyle: {
          fontWeight: 'bold',
          fontFamily: 'Outfit-Bold',
        },
        contentStyle: {
          backgroundColor: '#0F172A', // Slate 900
        },
      }}
    >
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{ title: 'FaceAuth SDK Hub' }}
      />
      <Stack.Screen
        name="Enrollment"
        component={EnrollmentScreen}
        options={{ title: 'Face Enrollment' }}
      />
      <Stack.Screen
        name="Authentication"
        component={AuthenticationScreen}
        options={{ title: 'Biometric Login' }}
      />
      <Stack.Screen
        name="VerificationResult"
        component={VerificationResultScreen}
        options={{ title: 'Access Verdict' }}
      />
      <Stack.Screen
        name="RegisteredUsers"
        component={RegisteredUsersScreen}
        options={{ title: 'User Registry' }}
      />
      <Stack.Screen
        name="SyncDashboard"
        component={SyncDashboardScreen}
        options={{ title: 'Cloud Sync Console' }}
      />
      <Stack.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ title: 'Engine Settings' }}
      />
    </Stack.Navigator>
  );
}
