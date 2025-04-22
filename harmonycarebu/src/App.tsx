import React, {  createRef } from 'react'

import { MessageBoxRight, MessageBoxLeft } from './components/MessageBox'
import MessageLoading from './components/MessageLoading'

import './App.css'

const chatboxRef  = createRef() as React.RefObject<HTMLDivElement>

let canSendMessage = true

class App extends React.Component {
    constructor(props: {}) {
        super(props);

        this.onSubmit = this.onSubmit.bind(this)
    }

    createSystemMessage(query='', image='') {
        const ele = document.createElement("div")
        ele.className = 'flex-start'

        const systemMessageBox = MessageBoxLeft({ content: '' })
        systemMessageBox.appendChild(MessageLoading())
        
        ele.appendChild(systemMessageBox) 

        chatboxRef.current.appendChild(ele)

        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, image }),
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then((data) => {
            const answer = data.answer

            systemMessageBox.textContent  = answer

            if (typeof data.image === 'string' && data.image.trim().length > 0) {
                const img = document.createElement('img')
                img.src = 'data:image/png;base64,' + data.image
                systemMessageBox.appendChild(document.createElement("br"))
                systemMessageBox.appendChild(img)
            }
            const button = document.createElement('button')
            systemMessageBox.appendChild(document.createElement("br"))
            systemMessageBox.appendChild(button)
            button.textContent = '朗讀(read)'

        })
        .catch((_error) => {
            systemMessageBox.textContent  = '❌ 連線錯誤，請稍後再試。'
        })
        .finally(() => {
            canSendMessage = true
        })
    }

    onSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault()

        if (!canSendMessage) {
            return
        }

        canSendMessage = false

        const form = e.currentTarget
        const formElements = form.elements as typeof form.elements & {
            query: HTMLInputElement,
            image: HTMLInputElement
        }

        const ele = document.createElement("div")
        ele.className = 'flex-end'
        
        const messageBox = MessageBoxRight({ content: formElements.query.value || '' })
        ele.appendChild(messageBox)

        let base64Image = ''

        if(formElements.image.files?.length) {
            const file = formElements.image.files[0];
            const reader = new FileReader()
            reader.onload = (e) => {
                if (!e.target || !e.target.result) return

                base64Image = e.target.result.toString().split(",")[1];

                const img = document.createElement('img')
                img.src = e.target.result as string

                messageBox.appendChild(document.createElement('br'))
                messageBox.appendChild(img)
            }
            reader.readAsDataURL(file)
        }

        chatboxRef.current.appendChild(ele)

        const queryText = formElements.query.value

        formElements.query.value = ''
        formElements.image.value = ''
        document.getElementById('image-filename')!.innerHTML = ''

        setTimeout(() => {
            this.createSystemMessage(queryText, base64Image)
        }, 500)
    }

    onFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
        e.preventDefault()

        const input = e.currentTarget
        const files = input.files
        
        const filename = files && files[0]?.['name']

        document.getElementById('image-filename')!.innerText = filename || ''
    }

    render()  {
        return (
        <>
            <div className='full-screen'>
                <div className='header-box'>
                    <div className='title-box'>HarmonyCare</div>
                </div>
                <div className="chat-box" ref={ chatboxRef }>
                </div>
                <div className='input-box '>
                    <form onSubmit={ this.onSubmit }>
                        <div className='user-input-box'>
                                <input className='user-input' id='query' type='text' placeholder='詢問任何問題！' />
                                <label htmlFor='image' className='image-upload'> 
                                    <span className='material-symbols-outlined'>image</span>
                                    <span id='image-filename' className='image-filename' ></span>
                                </label>
                                <input type='file' id='image' name='image' accept='image/*' onChange={ this.onFileUpload }/>
                        </div>
                    </form>
                </div>
            </div>
        </>
        )
    }
}

export default App
