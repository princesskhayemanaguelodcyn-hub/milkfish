package com.example.bangusfreshness

import android.content.Context
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia
import androidx.activity.result.contract.ActivityResultContracts.PickVisualMedia.ImageAndVideo
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.example.bangusfreshness.api.ClassifyResponse
import com.example.bangusfreshness.api.apiClient
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    BangusScreen()
                }
            }
        }
    }
}

@Composable
fun BangusScreen() {
    val context = LocalContext.current
    var serverUrl by remember { mutableStateOf("http://10.0.2.2:8000") }
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var result by remember { mutableStateOf<ClassifyResponse?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    val pickMediaLauncher = rememberLauncherForActivityResult(PickVisualMedia()) { uri ->
        imageUri = uri
    }

    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(text = "Bangus Freshness", style = MaterialTheme.typography.headlineSmall)

        OutlinedTextField(
            value = serverUrl,
            onValueChange = { serverUrl = it },
            label = { Text("Server URL") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
            keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = androidx.compose.foundation.text.KeyboardActions(onDone = {})
        )

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = {
                pickMediaLauncher.launch(PickVisualMediaRequest(ImageAndVideo))
            }) { Text("Pick image") }

            Button(
                enabled = imageUri != null && !isLoading,
                onClick = {
                    val uri = imageUri ?: return@Button
                    scope.launch {
                        isLoading = true
                        error = null
                        result = null
                        try {
                            val file = copyUriToTempFile(context, uri)
                            val client = apiClient(serverUrl)
                            val resp = client.classify(file)
                            result = resp
                        } catch (e: Exception) {
                            error = e.message
                        } finally {
                            isLoading = false
                        }
                    }
                }
            ) { Text("Classify") }
        }

        imageUri?.let { uri ->
            AsyncImage(
                model = ImageRequest.Builder(context).data(uri).crossfade(true).build(),
                contentDescription = null,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp)
            )
        }

        if (isLoading) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                CircularProgressIndicator()
            }
        }

        error?.let { Text(text = it, color = MaterialTheme.colorScheme.error) }

        result?.let { r ->
            Text(text = "Result: ${r.label}")
            Text(text = "Confidence: ${"%.2f".format(r.confidence)}")
            Text(text = "Eye detected: ${r.eye_detected}")
            Text(text = "Whiteness: ${"%.2f".format(r.metrics.whiteness_ratio)}  Yellowness: ${"%.2f".format(r.metrics.yellowness_ratio)}  Darkness: ${"%.2f".format(r.metrics.darkness_ratio)}")
        }
    }
}

private fun copyUriToTempFile(context: Context, uri: Uri): File {
    val input = context.contentResolver.openInputStream(uri) ?: error("Cannot open input stream")
    val temp = File.createTempFile("bangus", ".img", context.cacheDir)
    FileOutputStream(temp).use { out ->
        input.copyTo(out)
    }
    return temp
}

@Preview(showBackground = true)
@Composable
fun BangusScreenPreview() {
    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize()) {
            BangusScreen()
        }
    }
}