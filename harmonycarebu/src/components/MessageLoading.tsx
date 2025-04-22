import './MessageLoading.css'

const MessageLoading = () => {
    const ele = document.createElement("span")
    ele.className = 'dots-cont'

    const dot1 = document.createElement("span")
    dot1.className = 'dot dot-1'

    const dot2 = document.createElement("span")
    dot2.className = 'dot dot-2'

    const dot3 = document.createElement("span")
    dot3.className = 'dot dot-3'

    ele.appendChild(dot1)
    ele.appendChild(dot2)
    ele.appendChild(dot3)

    return ele
}

export default MessageLoading