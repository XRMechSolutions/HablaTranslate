# Natives on WhisperNative are bound by JNI symbol name.
-keepclasseswithmembernames class * { native <methods>; }
# NativeResult is constructed only from JNI (constructor id cached in JNI_OnLoad).
-keep class com.xrmech.habla.asr.internal.NativeResult { <init>(...); }
