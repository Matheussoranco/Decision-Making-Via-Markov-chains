import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class MathProblemSolver:
    def __init__(self):
        """Inicializa o resolvedor de problemas matemáticos com ML"""
        self.tokenizer = Tokenizer()
        self.max_sequence_len = 0
        self.model = None
        self.strategy_model = None
        self.problem_types = {}
        self.solution_steps = {}
        
    def preprocess_data(self, problems: List[str], solutions: List[List[str]]):
        """Preprocessa os dados de problemas e soluções"""
        # Tokeniza os problemas
        self.tokenizer.fit_on_texts(problems)
        sequences = self.tokenizer.texts_to_sequences(problems)
        self.max_sequence_len = max(len(seq) for seq in sequences)
        
        # Cria vocabulário para passos da solução
        all_steps = [step for solution in solutions for step in solution]
        unique_steps = list(set(all_steps))
        self.step_to_idx = {step: i for i, step in enumerate(unique_steps)}
        self.idx_to_step = {i: step for i, step in enumerate(unique_steps)}
        
        # Prepara dados de treinamento
        X = pad_sequences(sequences, maxlen=self.max_sequence_len, padding='post')
        y = []
        for solution in solutions:
            step_indices = [self.step_to_idx[step] for step in solution]
            y.append(step_indices)
        
        return X, y
    
    def build_models(self, vocab_size: int, num_steps: int):
        """Constrói os modelos de ML"""
        # Modelo principal para prever passos da solução
        self.model = Sequential([
            Embedding(input_dim=vocab_size+1, output_dim=64, input_length=self.max_sequence_len),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(64, activation='relu'),
            Dense(num_steps, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
        # Modelo auxiliar para classificar o tipo de problema
        self.strategy_model = Sequential([
            Embedding(input_dim=vocab_size+1, output_dim=64, input_length=self.max_sequence_len),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(len(self.problem_types), activation='softmax')
        ])
        
        self.strategy_model.compile(optimizer='adam',
                                   loss='sparse_categorical_crossentropy',
                                   metrics=['accuracy'])
    
    def train(self, problems: List[str], solutions: List[List[str]], problem_types: List[str]):
        """Treina o modelo com problemas e soluções"""
        # Mapeia tipos de problemas
        unique_types = list(set(problem_types))
        self.problem_types = {pt: i for i, pt in enumerate(unique_types)}
        self.type_to_name = {i: pt for i, pt in enumerate(unique_types)}
        
        # Preprocessa dados
        X, y_steps = self.preprocess_data(problems, solutions)
        y_types = np.array([self.problem_types[pt] for pt in problem_types])
        
        # Converte y_steps para formato adequado
        max_steps = max(len(solution) for solution in solutions)
        y_steps_padded = pad_sequences(y_steps, maxlen=max_steps, padding='post', value=-1)
        
        # Divide dados de treinamento e teste
        X_train, X_test, yt_train, yt_test, ys_train, ys_test = train_test_split(
            X, y_types, y_steps_padded, test_size=0.2, random_state=42
        )
        
        # Treina modelo de classificação de tipo
        print("\nTreinando classificador de tipos de problemas...")
        self.strategy_model.fit(X_train, yt_train, epochs=20, batch_size=32, 
                               validation_data=(X_test, yt_test))
        
        # Treina modelo de passos da solução
        print("\nTreinando modelo de passos da solução...")
        history = self.model.fit(
            np.repeat(X_train, max_steps, axis=0),
            ys_train.flatten(),
            epochs=30,
            batch_size=32,
            validation_split=0.1
        )
        
        # Plota histórico de treinamento
        plt.plot(history.history['accuracy'], label='Acurácia')
        plt.plot(history.history['loss'], label='Perda')
        plt.title('Desempenho do Modelo')
        plt.legend()
        plt.show()
    
    def solve(self, problem: str, max_steps: int = 10) -> List[str]:
        """Resolve um problema matemático"""
        # Preprocessa o problema
        sequence = self.tokenizer.texts_to_sequences([problem])
        padded_seq = pad_sequences(sequence, maxlen=self.max_sequence_len, padding='post')
        
        # Classifica o tipo de problema
        prob_type_idx = np.argmax(self.strategy_model.predict(padded_seq))
        prob_type = self.type_to_name[prob_type_idx]
        print(f"Tipo de problema identificado: {prob_type}")
        
        # Gera passos da solução
        solution_steps = []
        for _ in range(max_steps):
            pred = self.model.predict(padded_seq)
            next_step_idx = np.argmax(pred)
            next_step = self.idx_to_step[next_step_idx]
            
            if next_step == 'SOLUCAO_FINAL':
                break
                
            solution_steps.append(next_step)
            # Atualiza a sequência com o passo atual (simulando memória)
            sequence[0].append(next_step_idx)
            padded_seq = pad_sequences(sequence, maxlen=self.max_sequence_len, padding='post')
        
        return solution_steps
    
    def explain_solution(self, problem: str):
        """Resolve e explica o problema passo a passo"""
        solution = self.solve(problem)
        print(f"\nProblema: {problem}")
        print("\nProcesso de Solução:")
        for i, step in enumerate(solution, 1):
            print(f"{i}. {step}")
        print("\nSolução final encontrada!")