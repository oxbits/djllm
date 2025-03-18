import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { Button, StyleSheet, Text, View, TextInput } from 'react-native';

export default function App() {

  const [response, setResponse] = useState('')

  const [entry, setEntry] = useState('')

  const get_djllm = () => {

    fetch('http://127.0.0.1:8753/djllm?' + entry
    ).then(
      (resp) => { 
      resp.json().then((data) => {
          console.log(data);
          setResponse(data.response)
      }).catch((err) => {
          console.log(err);
      })
    })
  }

  return (
    <View style={styles.container}>
      <TextInput
          style={styles.input}
          onChangeText={setEntry}
          value={entry}
          placeholder="entry"
        />
        <Button
          onPress={get_djllm}
          title="submit"
          color="#841584"
          accessibilityLabel="submit entry to djllm"
        />
        <View style={{height: 20}}/>
      <Text>RESPONSE:</Text>
      <View style={{height: 10}}/>
      <Text>{response}</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  input: {
    height: 40,
    margin: 12,
    borderWidth: 1,
    padding: 10,
    width: 400,
  },
});
