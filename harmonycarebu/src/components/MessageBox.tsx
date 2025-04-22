import './MessageBox.css'

interface MessageBoxProps {
    content: string
}

function MessageBoxRight(props: MessageBoxProps) {
    const ele = document.createElement("div");
    ele.className = 'message-box message-box-right'
    ele.textContent = props.content

    return ele;
}

function MessageBoxLeft(props: MessageBoxProps) {
    const ele = document.createElement("div");
    ele.className = 'message-box message-box-left'
    ele.textContent = props.content

    return ele
}

export {
    MessageBoxRight,
    MessageBoxLeft,
}
