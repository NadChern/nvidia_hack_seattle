package com.visualmemory.glasses

import android.app.Application
import com.ffalcon.mercury.android.sdk.MercurySDK

class VisualMemoryApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        MercurySDK.init(this)
    }
}
