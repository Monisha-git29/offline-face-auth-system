import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, TextInput, FlatList, TouchableOpacity, Alert, SafeAreaView, ActivityIndicator } from 'react-native';
import BiometricService, { UserProfile } from '../services/BiometricService';

export default function RegisteredUsersScreen() {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const userList = await BiometricService.getUsers();
      setUsers(userList);
    } catch (e) {
      console.error(e);
      Alert.alert("Database Error", "Failed to retrieve user registry records.");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = (userId: string, name: string) => {
    Alert.alert(
      "Confirm Deletion",
      `Are you sure you want to delete profile registry for ${name} (ID: ${userId})? This action is offline-permanent.`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              const success = await BiometricService.deleteUser(userId);
              if (success) {
                Alert.alert("Deleted", "User deleted successfully.");
                fetchUsers();
              }
            } catch (err: any) {
              Alert.alert("Delete Failed", err.message || "Failed to remove user record.");
            }
          }
        }
      ]
    );
  };

  const filteredUsers = users.filter(user => 
    user.userId.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <TextInput
          style={styles.searchBar}
          value={searchQuery}
          onChangeText={setSearchQuery}
          placeholder="Search users by name or ID..."
          placeholderTextColor="#64748B"
        />

        {loading ? (
          <ActivityIndicator size="large" color="#3B82F6" style={{ marginTop: 40 }} />
        ) : filteredUsers.length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>
              {searchQuery ? "No matching records found." : "No users enrolled in local database."}
            </Text>
          </View>
        ) : (
          <FlatList
            data={filteredUsers}
            keyExtractor={(item) => item.userId}
            contentContainerStyle={styles.listContainer}
            renderItem={({ item }) => (
              <View style={styles.userCard}>
                <View style={styles.userCardInfo}>
                  <Text style={styles.userName}>{item.name}</Text>
                  <Text style={styles.userId}>ID: {item.userId}</Text>
                  <Text style={styles.userDate}>
                    Registered: {new Date(item.createdAt).toLocaleDateString()}
                  </Text>
                </View>
                <TouchableOpacity
                  style={styles.deleteBtn}
                  onPress={() => handleDelete(item.userId, item.name)}
                >
                  <Text style={styles.deleteBtnText}>Delete</Text>
                </TouchableOpacity>
              </View>
            )}
          />
        )}
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
    flex: 1,
    padding: 24,
  },
  searchBar: {
    backgroundColor: '#1E293B',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: '#F8FAFC',
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 20,
    fontSize: 15,
    fontFamily: 'Outfit-Regular',
  },
  listContainer: {
    gap: 12,
  },
  userCard: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: '#334155',
  },
  userCardInfo: {
    flex: 1,
  },
  userName: {
    fontSize: 18,
    fontWeight: '700',
    color: '#F8FAFC',
    fontFamily: 'Outfit-Bold',
  },
  userId: {
    fontSize: 13,
    color: '#3B82F6',
    fontWeight: '600',
    marginTop: 2,
    fontFamily: 'Outfit-Medium',
  },
  userDate: {
    fontSize: 11,
    color: '#64748B',
    marginTop: 4,
    fontFamily: 'Outfit-Regular',
  },
  deleteBtn: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    borderColor: '#EF4444',
    borderWidth: 1,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  deleteBtnText: {
    color: '#EF4444',
    fontWeight: '700',
    fontSize: 12,
    fontFamily: 'Outfit-Bold',
  },
  emptyContainer: {
    alignItems: 'center',
    marginTop: 40,
  },
  emptyText: {
    color: '#64748B',
    fontSize: 15,
    fontFamily: 'Outfit-Regular',
  },
});
