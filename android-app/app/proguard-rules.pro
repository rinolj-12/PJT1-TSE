# ONNX Runtime: Java classes are created and called from JNI
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }
# the app's own classes (small)
-keep class edu.vit.tsedemo.** { *; }
