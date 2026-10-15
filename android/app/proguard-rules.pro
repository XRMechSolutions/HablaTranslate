# ONNX Runtime reaches its Java classes from JNI by name.
-keep class ai.onnxruntime.** { *; }
# Ktor references SLF4J optionally.
-dontwarn org.slf4j.**
