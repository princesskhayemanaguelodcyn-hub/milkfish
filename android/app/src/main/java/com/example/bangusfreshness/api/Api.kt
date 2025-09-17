package com.example.bangusfreshness.api

import com.squareup.moshi.Json
import com.squareup.moshi.JsonClass
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

@JsonClass(generateAdapter = true)
data class Metrics(
    @Json(name = "whiteness_ratio") val whiteness_ratio: Double,
    @Json(name = "yellowness_ratio") val yellowness_ratio: Double,
    @Json(name = "darkness_ratio") val darkness_ratio: Double,
    @Json(name = "sharpness_score") val sharpness_score: Double,
)

@JsonClass(generateAdapter = true)
data class ClassifyResponse(
    val success: Boolean,
    val label: String,
    val confidence: Double,
    val metrics: Metrics,
    @Json(name = "eye_detected") val eye_detected: Boolean,
)

interface FreshnessApi {
    @Multipart
    @POST("/classify")
    suspend fun classify(@Part file: MultipartBody.Part): ClassifyResponse
}

fun apiClient(baseUrl: String): FreshnessApi {
    val logging = HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BODY }
    val client = OkHttpClient.Builder()
        .addInterceptor(logging)
        .build()

    val retrofit = Retrofit.Builder()
        .baseUrl(if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/")
        .addConverterFactory(MoshiConverterFactory.create())
        .client(client)
        .build()

    return retrofit.create(FreshnessApi::class.java)
}

suspend fun FreshnessApi.classify(file: java.io.File): ClassifyResponse {
    val requestFile = file.asRequestBody("application/octet-stream".toMediaType())
    val body = MultipartBody.Part.createFormData("file", file.name, requestFile)
    return classify(body)
}