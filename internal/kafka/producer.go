package kafka

import (
	"context"
	"errors"
	"log"
	"time"

	"github.com/segmentio/kafka-go"
)

// Producer Kafka-ya mesaj göndərmək üçün lazım olan konfiqurasiyanı saxlayır.
type Producer struct {
	writers map[string]*kafka.Writer // topic -> writer
	brokers []string
}

// NewProducer yeni bir Kafka producer yaradır və konfiqurasiya edir.
func NewProducer(brokers []string, topics ...string) (*Producer, error) {
	writers := make(map[string]*kafka.Writer)

	// Her topic üçün ayrı writer yaradırıq.
	for _, topic := range topics {
		w := &kafka.Writer{
			Addr:         kafka.TCP(brokers...),
			Topic:        topic,
			Balancer:     &kafka.LeastBytes{},
			WriteTimeout: 10 * time.Second,
			ReadTimeout:  10 * time.Second,
		}
		writers[topic] = w
		log.Printf("[Kafka] Writer created for topic '%s' with brokers: %v", topic, brokers)
	}

	return &Producer{
		writers: writers,
		brokers: brokers,
	}, nil
}

// SendMessage ilə default topic-ə mesaj göndərik (backward compatibility).
// Əgər topic name ilə gəlibsə, o topic-ə gönderək.
func (p *Producer) SendMessage(cameraIDOrKey, message string) error {
	// Əgər single argument gəlibsə, ilk topic-ə göndər (legacy).
	// Əgər iki argument gəlibsə, ikinci argument key-dir.

	if len(p.writers) == 0 {
		return errNoWriters
	}

	// İlk topic-ə göndər (default)
	var topic string
	for t := range p.writers {
		topic = t
		break
	}

	return p.SendMessageToTopic(topic, cameraIDOrKey, message)
}

// SendMessageToTopic spesifik topic-ə mesaj göndərir.
func (p *Producer) SendMessageToTopic(topic, key, message string) error {
	writer, ok := p.writers[topic]
	if !ok {
		log.Printf("[Kafka] Topic '%s' not found. Available topics: %v", topic, getTopicList(p.writers))
		return errTopicNotFound
	}

	msg := kafka.Message{
		Key:   []byte(key), // Camera ID partisyon key olarak istifadə olunur
		Value: []byte(message),
	}

	err := writer.WriteMessages(context.Background(), msg)
	if err != nil {
		log.Printf("[Kafka] ERROR: Failed to write message to topic '%s': %v", topic, err)
		return err
	}

	log.Printf("[Kafka] Message sent to topic '%s' (key: %s)", topic, key)
	return nil
}

// Close bütün Kafka writers-i bağlayır.
func (p *Producer) Close() error {
	for topic, writer := range p.writers {
		if err := writer.Close(); err != nil {
			log.Printf("[Kafka] ERROR: Failed to close writer for topic '%s': %v", topic, err)
			return err
		}
	}
	log.Println("[Kafka] All writers closed")
	return nil
}

// Utility funksiyonları
var (
	errNoWriters     = errors.New("no writers configured")
	errTopicNotFound = errors.New("topic not found")
)

func getTopicList(writers map[string]*kafka.Writer) []string {
	topics := make([]string, 0, len(writers))
	for topic := range writers {
		topics = append(topics, topic)
	}
	return topics
}
