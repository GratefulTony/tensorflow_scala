/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_platanios_tensorflow_jni_Tensor__ */

#ifndef _Included_org_platanios_tensorflow_jni_Tensor__
#define _Included_org_platanios_tensorflow_jni_Tensor__
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    allocate
 * Signature: (I[JJ)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_allocate
  (JNIEnv *, jobject, jint, jlongArray, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    fromBuffer
 * Signature: (I[JJL)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_fromBuffer
  (JNIEnv *, jobject, jint, jlongArray, jlong, jobject);

///*
// * Class:     org_platanios_tensorflow_jni_Tensor__
// * Method:    fromBuffer
// * Signature: (LI[JJ)J
// */
//JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_fromBuffer
//  (JNIEnv *, jobject, jobject, jint, jlongArray, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    dataType
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_dataType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    shape
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_shape
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    buffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_buffer
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_delete
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    getEncodedStringSize
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_getEncodedStringSize
  (JNIEnv *, jobject, jint);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    setStringBytes
 * Signature: ([BL)I
 */
JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_setStringBytes
  (JNIEnv *, jobject, jbyteArray, jobject);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    getStringBytes
 * Signature: (L)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_getStringBytes
  (JNIEnv *, jobject, jobject);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerAllocateContext
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocateContext
  (JNIEnv *, jobject, jbyteArray);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerDeleteContext
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDeleteContext
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerAllocate
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerAllocate
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerDataType
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDataType
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerShape
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerShape
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerDevice
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDevice
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerDelete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerDelete
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerResolve
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerResolve
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerCopyToDevice
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerCopyToDevice
  (JNIEnv *, jobject, jlong, jlong, jstring);

/*
 * Class:     org_platanios_tensorflow_jni_Tensor__
 * Method:    eagerSetOpDevice
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_platanios_tensorflow_jni_Tensor_00024_eagerSetOpDevice
  (JNIEnv *, jobject, jlong, jlong, jstring);

#ifdef __cplusplus
}
#endif
#endif
