export const BASE_URL = import.meta.env.DEV ? 'http://localhost:8000' : ''

export const get = async <T = undefined>(url: string) => {
    const response = await fetch(`${BASE_URL}${url}`)

    if (response.ok) {
        return await response.json() as T
    } else {
        return Promise.reject()
    }
}

export const post = async <T = undefined>(url: string, body?: unknown) => {
    const response = await fetch(`${BASE_URL}${url}`, {
        method: "POST",
        headers: {
            'Content-Type': 'application/json; charset=UTF-8' 
        },
        ...(body ? { body: JSON.stringify(body) } : {})
    })

    if (response.ok) {
        return await response.json() as T
    } else {
        return Promise.reject()
    }
}