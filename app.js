import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, Image, StyleSheet, Alert, ScrollView, TouchableOpacity, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function App() {
  const [mode, setMode] = useState('text'); // 'text' or 'image'
  const [breed1, setBreed1] = useState('');
  const [breed2, setBreed2] = useState('');
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [recognizedBreeds, setRecognizedBreeds] = useState(null);
  const [puppyImages, setPuppyImages] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Sorry, we need camera roll permissions to make this work!');
      }
    })();
  }, []);

  const pickImage = async (setImage) => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setRecognizedBreeds(null); // Reset recognized breeds when new image is picked
    }
  };

  const generatePuppy = async () => {
    if (loading) return;

    setLoading(true);
    setPuppyImages([]);
    setRecognizedBreeds(null);

    // Show loading message immediately
    Alert.alert(
      'Generating Puppies',
      'This may take ~60 seconds due to high-quality rendering. Please wait...',
      [{ text: 'OK', onPress: () => {} }],
      { cancelable: false }
    );

    try {
      let body;
      if (mode === 'image') {
        if (!image1 || !image2) {
          Alert.alert('Error', 'Please upload both dog images');
          setLoading(false);
          return;
        }

        const img1Response = await fetch(image1);
        const img1Data = await img1Response.blob();
        const img1Base64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result.split(',')[1]); // Remove data URI prefix
          reader.readAsDataURL(img1Data);
        });

        const img2Response = await fetch(image2);
        const img2Data = await img2Response.blob();
        const img2Base64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result.split(',')[1]); // Remove data URI prefix
          reader.readAsDataURL(img2Data);
        });

        body = JSON.stringify({
          mode: 'image',
          image1: img1Base64,
          image2: img2Base64,
        });
      } else {
        if (!breed1 || !breed2) {
          Alert.alert('Error', 'Please enter both breed names');
          setLoading(false);
          return;
        }
        body = JSON.stringify({
          mode: 'text',
          breed1: breed1.toLowerCase().replace(" ", "_"),
          breed2: breed2.toLowerCase().replace(" ", "_"),
        });
      }

      console.log('Sending initial request with mode:', mode);
      await fetch('YOUR NGROK URL/generate_puppy?init=true', { // Replace with your ngrok URL
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });

      // Poll for result
      let result = null;
      const maxAttempts = 12; // ~60 seconds with 5-second intervals
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 5000)); // Poll every 5 seconds
        console.log('Polling attempt', attempt + 1);
        const response = await fetch('YOUR NGROK URL/generate_puppy', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        });
        result = await response.json();
        console.log('Poll response:', result);
        if (result.error) throw new Error(result.error);
        if (result.images && result.images.length > 0) break;
      }

      if (!result || !result.images || result.images.length === 0) {
        throw new Error('Generation timed out or no images received');
      }

      // Process result
      setPuppyImages(result.images);
      if (mode === 'image' && result.recognized_breed1 && result.recognized_breed2) {
        const breed1Text = result.recognized_breed1.replace(/<\/?b>/g, ''); // Remove HTML tags
        const breed2Text = result.recognized_breed2.replace(/<\/?b>/g, ''); // Remove HTML tags
        setRecognizedBreeds(
          <Text>
            Recognized Breeds: <Text style={{ fontWeight: 'bold' }}>{breed1Text}</Text> and{' '}
            <Text style={{ fontWeight: 'bold' }}>{breed2Text}</Text>
          </Text>
        );
      }

    } catch (error) {
      console.error('Error generating puppies:', error.message);
      Alert.alert('Error', `Failed to generate images: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Puppy Generator</Text>
      
      {/* Mode Toggle */}
      <View style={styles.modeContainer}>
        <TouchableOpacity
          style={[styles.modeButton, mode === 'text' && styles.activeMode]}
          onPress={() => setMode('text')}
        >
          <Text style={styles.modeText}>Text to Puppy</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.modeButton, mode === 'image' && styles.activeMode]}
          onPress={() => setMode('image')}
        >
          <Text style={styles.modeText}>Image to Puppy</Text>
        </TouchableOpacity>
      </View>

      {/* Input Fields */}
      {mode === 'text' ? (
        <>
          <TextInput
            style={styles.input}
            placeholder="First breed (e.g., Labrador Retriever)"
            value={breed1}
            onChangeText={setBreed1}
          />
          <TextInput
            style={styles.input}
            placeholder="Second breed (e.g., Rottweiler)"
            value={breed2}
            onChangeText={setBreed2}
          />
        </>
      ) : (
        <>
          <TouchableOpacity
            style={styles.uploadButton}
            onPress={() => pickImage(setImage1)}
          >
            <Text style={styles.uploadText}>{image1 ? 'Change First Dog Image' : 'Upload First Dog Image'}</Text>
          </TouchableOpacity>
          {image1 && <Image source={{ uri: image1 }} style={styles.preview} />}
          <TouchableOpacity
            style={styles.uploadButton}
            onPress={() => pickImage(setImage2)}
          >
            <Text style={styles.uploadText}>{image2 ? 'Change Second Dog Image' : 'Upload Second Dog Image'}</Text>
          </TouchableOpacity>
          {image2 && <Image source={{ uri: image2 }} style={styles.preview} />}
          {recognizedBreeds && <Text style={styles.recognizedText}>{recognizedBreeds}</Text>}
        </>
      )}

      <Button
        title={loading ? 'Generating...' : 'Generate Puppy'}
        onPress={generatePuppy}
        disabled={loading || (mode === 'image' && (!image1 || !image2)) || (mode === 'text' && (!breed1 || !breed2))}
      />

      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loadingText}>Generating... (~60 seconds)</Text>
        </View>
      )}

      {puppyImages.length > 0 && (
        <View style={styles.imageContainer}>
          {puppyImages.map((item, index) => (
            <View key={index} style={styles.imageWrapper}>
              <Text style={styles.label}>{item.ratio}</Text>
              <Image
                source={{ uri: item.image_url }}
                style={styles.image}
                onError={(e) => console.log(`Image load error for ${item.ratio}:`, e.nativeEvent.error)}
              />
            </View>
          ))}
        </View>
      )}
      {!puppyImages.length && !loading && <Text style={styles.noImage}>No images to display</Text>}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: { fontSize: 28, fontWeight: 'bold', marginBottom: 20, color: '#333' },
  modeContainer: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  modeButton: {
    padding: 10,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    marginHorizontal: 5,
    backgroundColor: '#fff',
  },
  activeMode: {
    backgroundColor: '#4CAF50',
  },
  modeText: {
    fontSize: 16,
    color: '#333',
  },
  input: {
    width: '80%',
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 12,
    marginBottom: 15,
    borderRadius: 8,
    backgroundColor: '#fff',
  },
  uploadButton: {
    width: '80%',
    padding: 12,
    backgroundColor: '#4CAF50',
    borderRadius: 8,
    marginBottom: 15,
    alignItems: 'center',
  },
  uploadText: {
    color: '#fff',
    fontSize: 16,
  },
  preview: {
    width: 100,
    height: 100,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ddd',
    marginBottom: 15,
  },
  recognizedText: {
    fontSize: 16,
    color: '#333',
    marginBottom: 15,
  },
  imageContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  imageWrapper: {
    marginBottom: 15,
    alignItems: 'center',
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  image: {
    width: 300,
    height: 300,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  noImage: {
    fontSize: 16,
    color: '#888',
    marginTop: 20,
  },
  loadingContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: '#333',
    marginTop: 10,
  },
});