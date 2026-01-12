pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git 'https://github.com/AniketThings/NetFlix-Frontend.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t netflix-frontend-web:latest .'
            }
        }

        stage('Deploy with Docker Compose') {
            steps {
                sh 'docker compose down'
                sh 'docker compose up -d --build'
            }
        }
    }
}

