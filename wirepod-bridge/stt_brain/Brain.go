// Package wirepod_brain is a wire-pod STT "service" that bridges Vector's real
// microphone audio to the external Vector Brain server (our Python project),
// which runs a sensitive Vietnamese Whisper. This replaces wire-pod's built-in
// STT (which the user found not sensitive enough) while still using Vector's
// own mic captured by wire-pod.
//
// Set BRAIN_STT_URL to override the endpoint (default http://127.0.0.1:7070/stt).
package wirepod_brain

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"os"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/kercre123/wire-pod/chipper/pkg/logger"
	sr "github.com/kercre123/wire-pod/chipper/pkg/wirepod/speechrequest"
	"github.com/orcaman/writerseeker"
)

var Name string = "brain"

type brainResp struct {
	Text string `json:"text"`
}

func sttURL() string {
	if u := os.Getenv("BRAIN_STT_URL"); u != "" {
		return u
	}
	return "http://127.0.0.1:7070/stt"
}

func Init() error {
	logger.Println("Vector Brain STT bridge -> " + sttURL())
	return nil
}

// pcm2wav wraps raw 16 kHz / 16-bit / mono PCM in a WAV container.
func pcm2wav(in io.Reader) []byte {
	out := &writerseeker.WriterSeeker{}
	e := wav.NewEncoder(out, 16000, 16, 1, 1)
	audioBuf, err := newAudioIntBuffer(in)
	if err != nil {
		logger.Println(err)
	}
	if err := e.Write(audioBuf); err != nil {
		logger.Println(err)
	}
	if err := e.Close(); err != nil {
		logger.Println(err)
	}
	outBuf := new(bytes.Buffer)
	io.Copy(outBuf, out.BytesReader())
	return outBuf.Bytes()
}

func newAudioIntBuffer(r io.Reader) (*audio.IntBuffer, error) {
	buf := audio.IntBuffer{
		Format: &audio.Format{NumChannels: 1, SampleRate: 16000},
	}
	for {
		var sample int16
		err := binary.Read(r, binary.LittleEndian, &sample)
		switch {
		case err == io.EOF:
			return &buf, nil
		case err != nil:
			return nil, err
		}
		buf.Data = append(buf.Data, int(sample))
	}
}

// postToBrain sends the WAV to the Vector Brain server and returns the transcript.
func postToBrain(wavBytes []byte) string {
	httpReq, _ := http.NewRequest("POST", sttURL(), bytes.NewReader(wavBytes))
	httpReq.Header.Set("Content-Type", "audio/wav")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		logger.Println("brain STT request failed: " + err.Error())
		return ""
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	var r brainResp
	if err := json.Unmarshal(body, &r); err != nil {
		logger.Println("brain STT bad response: " + string(body))
		return ""
	}
	return r.Text
}

// STT collects the spoken audio (Vector's mic, via wire-pod VAD) and transcribes
// it through the Vector Brain server.
func STT(req sr.SpeechRequest) (string, error) {
	logger.Println("(Bot " + req.Device + ", Brain) Processing...")
	speechIsDone := false
	var err error
	for {
		_, err = req.GetNextStreamChunk()
		if err != nil {
			return "", err
		}
		speechIsDone, _ = req.DetectEndOfSpeech()
		if speechIsDone {
			break
		}
	}

	pcmBufTo := &writerseeker.WriterSeeker{}
	pcmBufTo.Write(req.DecodedMicData)
	pcmBuf := pcm2wav(pcmBufTo.BytesReader())

	transcribedText := postToBrain(pcmBuf)
	logger.Println("Bot " + req.Device + " Transcribed text: " + transcribedText)
	return transcribedText, nil
}
