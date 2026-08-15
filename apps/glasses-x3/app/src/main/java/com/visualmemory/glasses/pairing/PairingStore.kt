package com.visualmemory.glasses.pairing

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.visualmemory.glasses.model.DeviceCredential
import java.util.UUID
import kotlinx.coroutines.flow.first

private val Context.pairingDataStore by preferencesDataStore(name = "pairing")

class PairingStore(private val context: Context) {
    private object Keys {
        val gatewayUrl = stringPreferencesKey("gateway_url")
        val deviceId = stringPreferencesKey("device_id")
        val credential = stringPreferencesKey("credential")
        val credentialExpiresAt = stringPreferencesKey("credential_expires_at")
        val activeSessionId = stringPreferencesKey("active_session_id")
    }

    suspend fun deviceId(): String {
        val existing = context.pairingDataStore.data.first()[Keys.deviceId]
        if (existing != null) return existing
        val generated = "x3-${UUID.randomUUID()}"
        context.pairingDataStore.edit { it[Keys.deviceId] = generated }
        return generated
    }

    suspend fun load(): StoredPairing? {
        val values = context.pairingDataStore.data.first()
        val gatewayUrl = values[Keys.gatewayUrl] ?: return null
        val deviceId = values[Keys.deviceId] ?: return null
        val credential = values[Keys.credential] ?: return null
        val expiresAt = values[Keys.credentialExpiresAt] ?: return null
        return StoredPairing(
            gatewayUrl = gatewayUrl,
            credential = DeviceCredential(deviceId, credential, expiresAt),
        )
    }

    suspend fun activeSessionId(): String? =
        context.pairingDataStore.data.first()[Keys.activeSessionId]

    suspend fun saveActiveSessionId(sessionId: String?) {
        context.pairingDataStore.edit { values ->
            if (sessionId == null) values.remove(Keys.activeSessionId)
            else values[Keys.activeSessionId] = sessionId
        }
    }

    suspend fun save(gatewayUrl: String, credential: DeviceCredential) {
        context.pairingDataStore.edit { values ->
            values[Keys.gatewayUrl] = gatewayUrl
            values[Keys.deviceId] = credential.deviceId
            values[Keys.credential] = credential.credential
            values[Keys.credentialExpiresAt] = credential.expiresAt
        }
    }

    suspend fun clear() {
        val existingId = context.pairingDataStore.data.first()[Keys.deviceId]
            ?: "x3-${UUID.randomUUID()}"
        context.pairingDataStore.edit { values ->
            values.clear()
            values[Keys.deviceId] = existingId
        }
    }
}

data class StoredPairing(
    val gatewayUrl: String,
    val credential: DeviceCredential,
)
